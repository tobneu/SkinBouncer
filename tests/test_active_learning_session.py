from pathlib import Path

import pytest
from PIL import Image

from skinbouncer_core import evaluate_confusion_matrix, setup_detector_project, train_detector
from labeling_tool.active_learning_session import ActiveLearningSession, _reason, _suspicion


def _make_fixture_images(folder, prefix, n, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGBA", (64, 64), color).save(folder / f"{prefix}{i}.png")


def _make_trained_project(tmp_path, n_good=16, n_bad=16, ratios=(0.7, 0.15, 0.15)):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", n_good, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", n_bad, (200, 0, 0, 255))

    project_dir = tmp_path / "project"
    setup_detector_project(good_dir, bad_dir, project_dir, ratios=ratios)
    train_detector(project_dir, epochs=1, batch_size=8)
    return project_dir


# --- pure formula tests: no model, no fixtures, no TF ---

def test_suspicion_higher_for_good_with_higher_prob():
    assert _suspicion("good", 0.9) > _suspicion("good", 0.1)


def test_suspicion_higher_for_bad_with_lower_prob():
    assert _suspicion("bad", 0.1) > _suspicion("bad", 0.9)


def test_suspicion_symmetric_between_classes_at_same_prob():
    # a "good" image at p=0.8 and a "bad" image at p=0.2 are equally suspicious
    assert _suspicion("good", 0.8) == pytest.approx(_suspicion("bad", 0.2))


def test_reason_disagreement_when_predicted_class_differs():
    assert _reason("good", 0.9, threshold=0.5) == "model disagrees"
    assert _reason("bad", 0.1, threshold=0.5) == "model disagrees"


def test_reason_borderline_near_threshold_when_agreeing():
    assert _reason("good", 0.45, threshold=0.5) == "borderline"
    assert _reason("bad", 0.55, threshold=0.5) == "borderline"


def test_reason_confident_agreement_far_from_threshold():
    assert _reason("good", 0.05, threshold=0.5) == "confident agreement"
    assert _reason("bad", 0.95, threshold=0.5) == "confident agreement"


# --- real wiring smoke test: tiny trained checkpoint, no ranking-order assertions ---

def test_session_pools_train_and_val_excludes_test(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)

    manifest_test_filenames = {
        key.split("/", 1)[1]
        for key, info in session.manifest["images"].items()
        if info["split"] == "test"
    }
    shown_filenames = {item["path"].name for item in session.items}

    assert session.total() == len(session.items)
    assert session.total() > 0
    assert shown_filenames.isdisjoint(manifest_test_filenames)


def test_session_items_have_expected_shape(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)

    item = session.current_item()
    assert set(item) == {"key", "path", "recorded_class", "prob", "suspicion", "reason"}
    assert item["recorded_class"] in ("good", "bad")
    assert 0.0 <= item["prob"] <= 1.0


def test_session_has_confusion_matrix_after_init(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)

    assert session.confusion_matrix is not None
    assert set(session.confusion_matrix) == {"tp", "tn", "fp", "fn", "n"}
    assert session.confusion_matrix["n"] > 0


def test_session_raises_clear_error_if_not_a_project_dir(tmp_path):
    not_a_project = tmp_path / "just_some_images"
    _make_fixture_images(not_a_project, "img", 3, (0, 200, 0, 255))

    with pytest.raises(FileNotFoundError, match="not a detector project directory"):
        ActiveLearningSession(not_a_project)


def test_session_raises_if_project_never_trained(tmp_path):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", 4, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", 4, (200, 0, 0, 255))
    project_dir = tmp_path / "project"
    setup_detector_project(good_dir, bad_dir, project_dir)

    with pytest.raises(FileNotFoundError):
        ActiveLearningSession(project_dir)


def test_decide_skip_advances_without_relabeling(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)
    item_before = session.current_item()
    key_before = item_before["key"]

    session.decide("skip")

    assert session.index == 1
    assert session.relabel_count == 0
    assert key_before in session.manifest["images"]
    assert session.manifest["images"][key_before]["class"] == item_before["recorded_class"]


def test_decide_matching_recorded_class_is_confirmation_only(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)
    item = session.current_item()

    session.decide(item["recorded_class"])

    assert session.index == 1
    assert session.relabel_count == 0
    assert item["key"] in session.manifest["images"]


def test_decide_differing_action_relabels(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)
    item = session.current_item()
    new_class = "bad" if item["recorded_class"] == "good" else "good"
    old_key = item["key"]

    session.decide(new_class)

    assert session.index == 1
    assert session.relabel_count == 1
    assert old_key not in session.manifest["images"]
    new_key = f"{new_class}/{old_key.split('/', 1)[1]}"
    assert session.manifest["images"][new_key]["class"] == new_class


def test_decide_rejects_unknown_action(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)
    with pytest.raises(ValueError):
        session.decide("maybe")


def test_retrain_rebuilds_items_and_resets_index(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)
    session.decide("skip")
    session.decide("skip")
    relabel_count_before = session.relabel_count
    old_model_path_mtime = (project_dir / "model.keras").stat().st_mtime

    session.retrain(epochs=1, batch_size=8)
    session._retrain_thread.join()

    assert session.index == 0
    assert len(session.items) > 0
    assert session.relabel_count == relabel_count_before
    assert (project_dir / "model.keras").stat().st_mtime >= old_model_path_mtime


def test_retrain_runs_in_background_and_reports_progress(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)
    assert session.training_progress == {"status": "idle"}

    session.retrain(epochs=1, batch_size=8)
    session._retrain_thread.join()

    assert session.training_progress["status"] == "done"
    assert session.training_progress["epoch"] == 1
    assert session.training_progress["epochs_total"] == 1
    assert set(session.training_progress["history"]) >= {"auc", "val_auc"}
    assert len(session.training_progress["history"]["auc"]) == 1


def test_retrain_populates_run_comparison_against_previous_round(tmp_path):
    project_dir = _make_trained_project(tmp_path)  # already one training_runs.json entry
    session = ActiveLearningSession(project_dir)
    assert session.run_comparison is None

    session.retrain(epochs=1, batch_size=8)
    session._retrain_thread.join()

    assert session.run_comparison is not None
    assert "current" in session.run_comparison
    assert len(session.run_comparison["previous"]) == 1


def test_retrain_recomputes_confusion_matrix(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)

    session.retrain(epochs=1, batch_size=8)
    session._retrain_thread.join()

    assert session.confusion_matrix == evaluate_confusion_matrix(
        session.manifest, session.model, session.threshold
    )


def test_retrain_refuses_when_already_running(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)
    session.training_progress = {"status": "running"}

    with pytest.raises(RuntimeError, match="already in progress"):
        session.retrain(epochs=1, batch_size=8)


def test_retrain_error_path_sets_status_and_does_not_crash(tmp_path, monkeypatch):
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr("labeling_tool.active_learning_session.train_detector", _boom)

    session.retrain(epochs=1, batch_size=8)
    session._retrain_thread.join()

    assert session.training_progress["status"] == "error"
    assert session.training_progress["error"] == "simulated training failure"
    # session itself is still usable - unaffected by the failed retrain
    assert session.current_item() is not None


def test_decide_relabel_collision_leaves_session_unchanged(tmp_path):
    # Regression test: a real run hit this via the GUI - two different images
    # happened to share a filename across good/bad, so relabel_image's collision
    # guard correctly refused the move. The bug was downstream (app.js never reset
    # its "busy" flag on a rejected decide() promise, permanently freezing input).
    # That's only safe to unstick because the session's index does NOT advance when
    # relabel_image raises - this test locks that invariant in.
    project_dir = _make_trained_project(tmp_path)
    session = ActiveLearningSession(project_dir)
    item = session.current_item()
    old_key = item["key"]
    filename = old_key.split("/", 1)[1]
    new_class = "bad" if item["recorded_class"] == "good" else "good"

    colliding_dir = Path(session.manifest[f"{new_class}_dir"])
    Image.new("RGBA", (64, 64), (0, 0, 0, 255)).save(colliding_dir / filename)

    with pytest.raises(FileExistsError):
        session.decide(new_class)

    assert session.index == 0
    assert session.relabel_count == 0
    assert session.current_item()["key"] == old_key
    assert old_key in session.manifest["images"]
