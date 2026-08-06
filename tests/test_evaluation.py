import numpy as np
from PIL import Image

from skinbouncer_core import evaluate_confusion_matrix, load_model, setup_detector_project, train_detector


def _make_fixture_images(folder, prefix, n, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGBA", (64, 64), color).save(folder / f"{prefix}{i}.png")


class _StubModel:
    """Returns a fixed, caller-chosen probability per image regardless of pixel
    content - lets a test assert exact confusion-matrix counts without training a
    real model."""

    def __init__(self, probs):
        self._probs = np.array(probs, dtype=np.float32)

    def predict(self, X, verbose=0):
        assert len(X) == len(self._probs)
        return self._probs.reshape(-1, 1)


def _make_all_test_split_project(tmp_path, n_good, n_bad):
    # ratios=(0, 0, 1) forces every image into the frozen test split, so a test can
    # know exactly how many good/bad images to expect without depending on the
    # stratified shuffle's placement.
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", n_good, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", n_bad, (200, 0, 0, 255))
    project_dir = tmp_path / "project"
    manifest = setup_detector_project(good_dir, bad_dir, project_dir, ratios=(0.0, 0.0, 1.0))
    return manifest


def test_evaluate_confusion_matrix_counts_match_hand_computed_ground_truth(tmp_path):
    manifest = _make_all_test_split_project(tmp_path, n_good=4, n_bad=4)
    # _load_split_arrays orders test images good-then-bad, labels good=0/bad=1.
    probs = [0.1, 0.2, 0.6, 0.9, 0.1, 0.4, 0.6, 0.9]
    model = _StubModel(probs)

    cm = evaluate_confusion_matrix(manifest, model, threshold=0.5)

    assert cm == {"tp": 2, "tn": 2, "fp": 2, "fn": 2, "n": 8}
    assert all(isinstance(v, int) for v in cm.values())


def test_evaluate_confusion_matrix_returns_none_for_empty_test_split(tmp_path):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", 4, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", 4, (200, 0, 0, 255))
    project_dir = tmp_path / "project"
    manifest = setup_detector_project(good_dir, bad_dir, project_dir, ratios=(0.5, 0.5, 0.0))

    assert evaluate_confusion_matrix(manifest, model=None, threshold=0.5) is None


def test_evaluate_confusion_matrix_against_a_real_trained_checkpoint(tmp_path):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", 16, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", 16, (200, 0, 0, 255))
    project_dir = tmp_path / "project"
    manifest = setup_detector_project(good_dir, bad_dir, project_dir)
    train_detector(project_dir, epochs=1, batch_size=8)
    model = load_model(project_dir / "model.keras")
    threshold = 0.5

    cm = evaluate_confusion_matrix(manifest, model, threshold)

    assert set(cm) == {"tp", "tn", "fp", "fn", "n"}
    assert cm["n"] == cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
    assert cm["n"] > 0
    assert all(isinstance(v, int) for v in cm.values())
