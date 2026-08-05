import json
from unittest.mock import patch

from PIL import Image

from skinbouncer_core import load_model, setup_detector_project
from skinbouncer_core.train import _load_split_arrays, train_detector


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
