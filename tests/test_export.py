import json

import pytest
from PIL import Image

from skinbouncer_core import DEFAULT_DETECTORS_DIR, export_detector, setup_detector_project, train_detector


def _make_fixture_images(folder, prefix, n, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGBA", (64, 64), color).save(folder / f"{prefix}{i}.png")


def _make_trained_project(tmp_path, bad_folder_name="bad_demo"):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / bad_folder_name
    _make_fixture_images(good_dir, "good", 16, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", 16, (200, 0, 0, 255))
    project_dir = tmp_path / "project"
    setup_detector_project(good_dir, bad_dir, project_dir)
    train_detector(project_dir, epochs=1, batch_size=8)
    return project_dir


def _make_untrained_project(tmp_path):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", 4, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", 4, (200, 0, 0, 255))
    project_dir = tmp_path / "project"
    setup_detector_project(good_dir, bad_dir, project_dir)
    return project_dir


def test_export_writes_model_and_threshold_under_the_category_name(tmp_path):
    project_dir = _make_trained_project(tmp_path, bad_folder_name="spiderman")
    detectors_dir = tmp_path / "detectors"

    result = export_detector(project_dir, detectors_dir=detectors_dir)

    # The folder name is the manifest's category, not the project dir's name - it's
    # what the API reports each score under.
    assert result["category"] == "spiderman"
    dest = detectors_dir / "spiderman"
    assert (dest / "model.keras").exists()
    assert (dest / "threshold.json").exists()
    assert result["model_path"] == str(dest / "model.keras")
    assert result["threshold_path"] == str(dest / "threshold.json")
    assert result["dest_dir"] == str(dest)


def test_export_copies_the_threshold_verbatim(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    detectors_dir = tmp_path / "detectors"

    result = export_detector(project_dir, detectors_dir=detectors_dir)

    source_threshold = json.loads((project_dir / "threshold.json").read_text())["threshold"]
    exported = json.loads((detectors_dir / "bad_demo" / "threshold.json").read_text())
    assert exported["threshold"] == source_threshold
    assert result["threshold"] == source_threshold


def test_export_leaves_the_project_directory_untouched(tmp_path):
    project_dir = _make_trained_project(tmp_path)
    before = {p.name: p.stat().st_mtime for p in project_dir.iterdir() if p.is_file()}

    export_detector(project_dir, detectors_dir=tmp_path / "detectors")

    after = {p.name: p.stat().st_mtime for p in project_dir.iterdir() if p.is_file()}
    assert before == after


def test_export_overwrites_a_previous_export_of_the_same_category(tmp_path):
    # Re-exporting after another retrain round is the expected workflow, not an error.
    project_dir = _make_trained_project(tmp_path)
    detectors_dir = tmp_path / "detectors"
    export_detector(project_dir, detectors_dir=detectors_dir)

    stale = detectors_dir / "bad_demo" / "model.keras"
    stale.write_bytes(b"stale bytes from an older export")

    export_detector(project_dir, detectors_dir=detectors_dir)

    assert stale.read_bytes() == (project_dir / "model.keras").read_bytes()


def test_export_raises_when_the_project_was_never_trained(tmp_path):
    project_dir = _make_untrained_project(tmp_path)

    with pytest.raises(FileNotFoundError, match="no trained checkpoint to export"):
        export_detector(project_dir, detectors_dir=tmp_path / "detectors")


def test_export_raises_for_a_directory_that_is_not_a_project(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_detector(tmp_path / "nope", detectors_dir=tmp_path / "detectors")


def test_default_detectors_dir_is_the_folder_the_image_build_bakes_in():
    # Guards the export -> build.sh -> docker chain: build.sh reads exactly this path,
    # so a move of either side without the other silently breaks deployment.
    assert DEFAULT_DETECTORS_DIR.parts[-4:] == ("06_Deployment", "api", "models", "detectors")
    build_script = DEFAULT_DETECTORS_DIR.parents[3] / "06_Deployment" / "build.sh"
    assert 'DETECTORS_DIR="$SCRIPT_DIR/api/models/detectors"' in build_script.read_text()
