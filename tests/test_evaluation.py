import numpy as np
import pytest
from PIL import Image

from skinbouncer_core import (
    curation_status,
    evaluate_confusion_matrix,
    load_model,
    setup_detector_project,
    train_detector,
)

COUNT_KEYS = {"tp", "tn", "fp", "fn", "n"}
RATE_KEYS = {"recall", "precision", "accuracy"}


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

    assert {k: cm[k] for k in COUNT_KEYS} == {"tp": 2, "tn": 2, "fp": 2, "fn": 2, "n": 8}
    assert all(isinstance(cm[k], int) for k in COUNT_KEYS)
    # tp/(tp+fn), tp/(tp+fp) and (tp+tn)/n all land on 0.5 for this deliberately
    # symmetric case - each is checked separately below against an asymmetric one.
    assert cm["recall"] == pytest.approx(0.5)
    assert cm["precision"] == pytest.approx(0.5)
    assert cm["accuracy"] == pytest.approx(0.5)


def test_rates_distinguish_recall_precision_and_accuracy(tmp_path):
    # 6 good + 2 bad, so the three rates have three different denominators and a bug
    # swapping any two of them can't pass unnoticed.
    manifest = _make_all_test_split_project(tmp_path, n_good=6, n_bad=2)
    # good images first, then bad. Flags 3 of the 6 good (fp=3) and catches 1 of 2 bad.
    model = _StubModel([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9])

    cm = evaluate_confusion_matrix(manifest, model, threshold=0.5)

    assert {k: cm[k] for k in COUNT_KEYS} == {"tp": 1, "tn": 3, "fp": 3, "fn": 1, "n": 8}
    assert cm["recall"] == pytest.approx(1 / 2)
    assert cm["precision"] == pytest.approx(1 / 4)
    assert cm["accuracy"] == pytest.approx(4 / 8)


def test_rates_are_none_when_their_denominator_is_empty(tmp_path):
    # No bad images at all: "what share of bad images did it catch" has no answer, and
    # a model that flags nothing has never been right or wrong about a flag either.
    manifest = _make_all_test_split_project(tmp_path, n_good=4, n_bad=0)
    model = _StubModel([0.1, 0.1, 0.1, 0.1])

    cm = evaluate_confusion_matrix(manifest, model, threshold=0.5)

    assert cm["recall"] is None
    assert cm["precision"] is None
    assert cm["accuracy"] == pytest.approx(1.0)


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

    assert set(cm) == COUNT_KEYS | RATE_KEYS
    assert cm["n"] == cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
    assert cm["n"] > 0
    assert all(isinstance(cm[k], int) for k in COUNT_KEYS)
    assert all(cm[k] is None or 0.0 <= cm[k] <= 1.0 for k in RATE_KEYS)


def test_curation_status_counts_only_reviewed_test_entries(tmp_path):
    manifest = _make_all_test_split_project(tmp_path, n_good=3, n_bad=3)
    assert curation_status(manifest) == {"reviewed": 0, "total": 6, "complete": False}

    keys = list(manifest["images"])
    manifest["images"][keys[0]]["reviewed"] = True
    assert curation_status(manifest) == {"reviewed": 1, "total": 6, "complete": False}

    for key in keys:
        manifest["images"][key]["reviewed"] = True
    assert curation_status(manifest) == {"reviewed": 6, "total": 6, "complete": True}


def test_curation_status_is_not_complete_for_an_empty_test_split(tmp_path):
    # An empty test split has nothing to confirm, but reporting it as "curated" would
    # imply a held-out check happened that never did.
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", 4, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", 4, (200, 0, 0, 255))
    manifest = setup_detector_project(good_dir, bad_dir, tmp_path / "project", ratios=(0.5, 0.5, 0.0))

    assert curation_status(manifest) == {"reviewed": 0, "total": 0, "complete": False}
