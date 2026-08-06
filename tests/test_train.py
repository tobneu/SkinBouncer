import json
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from skinbouncer_core import load_model, setup_detector_project
from skinbouncer_core.train import (
    _compute_sample_weights,
    _load_split_arrays,
    _write_json,
    train_detector,
)


def _make_fixture_images(folder, prefix, n, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGBA", (64, 64), color).save(folder / f"{prefix}{i}.png")


def _make_project(tmp_path, n_good=16, n_bad=16):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", n_good, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", n_bad, (200, 0, 0, 255))

    project_dir = tmp_path / "project"
    setup_detector_project(good_dir, bad_dir, project_dir)
    return project_dir


def test_train_detector_writes_all_outputs(tmp_path):
    project_dir = _make_project(tmp_path)

    result = train_detector(project_dir, epochs=1, batch_size=8)

    assert result["model_path"] == project_dir / "model.keras"
    assert result["model_path"].exists()
    load_model(result["model_path"])  # must reload cleanly through the shared wrapper

    threshold_data = json.loads(result["threshold_path"].read_text())
    assert isinstance(threshold_data["threshold"], float)

    metrics_data = json.loads(result["metrics_path"].read_text())
    assert metrics_data["epochs_run"] == 1
    assert set(metrics_data["train"]) >= {"loss", "accuracy", "precision", "recall", "auc"}
    assert set(metrics_data["val"]) >= {"loss", "accuracy", "precision", "recall", "auc"}
    assert "threshold_search" in metrics_data

    assert set(result) >= {
        "model_path", "threshold_path", "metrics_path", "history",
        "epochs_run", "val_metrics", "threshold_info", "threshold_error",
    }


def test_train_detector_threshold_fallback_on_unreachable_recall(tmp_path):
    project_dir = _make_project(tmp_path)

    with patch("skinbouncer_core.train.find_threshold_for_recall", side_effect=ValueError("no threshold reaches target")):
        result = train_detector(project_dir, epochs=1, batch_size=8)

    assert result["threshold_error"] == "no threshold reaches target"
    assert result["threshold_info"]["threshold"] == 0.5
    assert result["model_path"].exists()

    threshold_data = json.loads(result["threshold_path"].read_text())
    assert threshold_data == {"threshold": 0.5}


def test_load_split_arrays_labels_good_0_bad_1(tmp_path):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", 3, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", 2, (200, 0, 0, 255))

    project_dir = tmp_path / "project"
    manifest = setup_detector_project(good_dir, bad_dir, project_dir, ratios=(1.0, 0.0, 0.0))

    X, y = _load_split_arrays(manifest, "train")
    assert X.shape[0] == 5
    assert list(y) == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_compute_sample_weights_is_inverse_frequency_per_class():
    # 3 good (label 0), 1 bad (label 1) -> bad should get the larger weight
    y_train = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    weights = _compute_sample_weights(y_train)

    good_weight = weights[y_train == 0][0]
    bad_weight = weights[y_train == 1][0]
    assert np.allclose(weights[y_train == 0], good_weight)  # same weight within a class
    assert bad_weight > good_weight  # minority class weighted higher
    assert good_weight == pytest.approx(4 / (2 * 3))
    assert bad_weight == pytest.approx(4 / (2 * 1))


def test_compute_sample_weights_handles_single_class_without_div_by_zero():
    y_train = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    weights = _compute_sample_weights(y_train)  # must not raise ZeroDivisionError
    assert len(weights) == 3


def test_train_detector_warm_start_loads_existing_model_instead_of_building_fresh(tmp_path):
    project_dir = _make_project(tmp_path)
    train_detector(project_dir, epochs=1, batch_size=8)

    with patch("skinbouncer_core.train.build_cnn") as mock_build_cnn:
        result = train_detector(project_dir, epochs=1, batch_size=8, warm_start=True)

    mock_build_cnn.assert_not_called()
    assert result["model_path"].exists()


def test_train_detector_warm_start_reuses_previous_recall_target_by_default(tmp_path):
    project_dir = _make_project(tmp_path)
    train_detector(project_dir, epochs=1, batch_size=8, recall_target=0.8)

    result = train_detector(project_dir, epochs=1, batch_size=8, warm_start=True)

    assert result["metrics_path"].exists()
    metrics = json.loads(result["metrics_path"].read_text())
    assert metrics["threshold_search"]["recall_target"] == pytest.approx(0.8)


def test_train_detector_warm_start_explicit_recall_target_overrides_reuse(tmp_path):
    project_dir = _make_project(tmp_path)
    train_detector(project_dir, epochs=1, batch_size=8, recall_target=0.8)

    result = train_detector(project_dir, epochs=1, batch_size=8, warm_start=True, recall_target=0.6)

    metrics = json.loads(result["metrics_path"].read_text())
    assert metrics["threshold_search"]["recall_target"] == pytest.approx(0.6)


def test_write_json_round_trips(tmp_path):
    path = tmp_path / "out.json"
    _write_json(path, {"threshold": 0.5})
    assert json.loads(path.read_text()) == {"threshold": 0.5}


def test_write_json_leaves_no_tmp_file_behind_on_success(tmp_path):
    path = tmp_path / "out.json"
    _write_json(path, {"a": 1})
    leftover = [p for p in tmp_path.iterdir() if p != path]
    assert leftover == []


def test_write_json_does_not_corrupt_existing_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "out.json"
    _write_json(path, {"threshold": 0.1})

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(json, "dump", _boom)
    with pytest.raises(RuntimeError):
        _write_json(path, {"threshold": 0.9})

    # original file must survive untouched, and no leftover temp file
    assert json.loads(path.read_text()) == {"threshold": 0.1}
    leftover = [p for p in tmp_path.iterdir() if p != path]
    assert leftover == []
