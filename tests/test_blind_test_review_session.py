import pytest
from PIL import Image

from labeling_tool.blind_test_review_session import BlindTestReviewSession
from skinbouncer_core import load_manifest, setup_detector_project


def _make_fixture_images(folder, prefix, n, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGBA", (64, 64), color).save(folder / f"{prefix}{i}.png")


def _make_project(tmp_path, n_good=6, n_bad=6, ratios=(0.0, 0.0, 1.0)):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", n_good, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", n_bad, (200, 0, 0, 255))

    project_dir = tmp_path / "project"
    setup_detector_project(good_dir, bad_dir, project_dir, ratios=ratios)
    return project_dir


def test_session_raises_clear_error_if_not_a_project_dir(tmp_path):
    not_a_project = tmp_path / "just_some_images"
    _make_fixture_images(not_a_project, "img", 3, (0, 200, 0, 255))
    with pytest.raises(FileNotFoundError, match="not a detector project directory"):
        BlindTestReviewSession(not_a_project)


def test_session_only_pools_test_split_images(tmp_path):
    # 70/15/15 split, mostly train/val - only the test slice should ever surface here.
    project_dir = _make_project(tmp_path, n_good=20, n_bad=20, ratios=(0.7, 0.15, 0.15))
    session = BlindTestReviewSession(project_dir)
    manifest = load_manifest(project_dir)

    assert len(session.items) > 0
    for item in session.items:
        assert manifest["images"][item["key"]]["split"] == "test"


def test_decide_matching_action_marks_reviewed_without_moving_file(tmp_path):
    project_dir = _make_project(tmp_path)
    session = BlindTestReviewSession(project_dir)
    item = session.current_item()
    old_path = item["path"]

    session.decide(item["recorded_class"])

    manifest = load_manifest(project_dir)
    assert manifest["images"][item["key"]]["reviewed"] is True
    assert manifest["images"][item["key"]]["class"] == item["recorded_class"]
    assert old_path.exists()


def test_decide_differing_action_relabels_and_marks_reviewed(tmp_path):
    project_dir = _make_project(tmp_path)
    session = BlindTestReviewSession(project_dir)
    item = session.current_item()
    old_path = item["path"]
    new_class = "bad" if item["recorded_class"] == "good" else "good"

    session.decide(new_class)

    manifest = load_manifest(project_dir)
    new_key = f"{new_class}/{item['key'].split('/', 1)[1]}"
    assert item["key"] not in manifest["images"]
    assert manifest["images"][new_key] == {"class": new_class, "split": "test", "reviewed": True}
    assert not old_path.exists()


def test_decide_rejects_skip(tmp_path):
    project_dir = _make_project(tmp_path)
    session = BlindTestReviewSession(project_dir)
    with pytest.raises(ValueError):
        session.decide("skip")


def test_progress_resumes_across_relaunches(tmp_path):
    project_dir = _make_project(tmp_path, n_good=2, n_bad=2)  # 4 test images total
    session1 = BlindTestReviewSession(project_dir)
    assert session1.total() == 4
    assert session1.index == 0

    reviewed_keys = []
    for _ in range(2):
        item = session1.current_item()
        reviewed_keys.append(item["key"])
        session1.decide(item["recorded_class"])

    session2 = BlindTestReviewSession(project_dir)
    assert session2.total() == 4  # frozen size unchanged
    assert session2.index == 2  # picks up where session1 left off
    assert session2.remaining() == 2
    assert len(session2.items) == 2
    assert all(item["key"] not in reviewed_keys for item in session2.items)


def test_is_done_and_total_after_reviewing_everything(tmp_path):
    project_dir = _make_project(tmp_path, n_good=1, n_bad=1)  # 2 test images total
    session = BlindTestReviewSession(project_dir)
    while not session.is_done():
        session.decide(session.current_item()["recorded_class"])

    assert session.is_done()
    assert session.total() == 2
    assert session.remaining() == 0

    # a fresh relaunch confirms nothing is left to review
    session2 = BlindTestReviewSession(project_dir)
    assert session2.is_done()
    assert session2.total() == 2
