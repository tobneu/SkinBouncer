import pytest
from PIL import Image

from skinbouncer_core import load_manifest, relabel_image, save_manifest, setup_detector_project


def _make_fixture_images(folder, prefix, n, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGBA", (64, 64), color).save(folder / f"{prefix}{i}.png")


def _make_project(tmp_path, n_good=10, n_bad=10, ratios=(1.0, 0.0, 0.0)):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", n_good, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", n_bad, (200, 0, 0, 255))

    project_dir = tmp_path / "project"
    manifest = setup_detector_project(good_dir, bad_dir, project_dir, ratios=ratios)
    return project_dir, manifest


def test_save_manifest_round_trips(tmp_path):
    project_dir, manifest = _make_project(tmp_path)
    manifest["category"] = "changed"
    save_manifest(manifest, project_dir)
    assert load_manifest(project_dir)["category"] == "changed"


def test_relabel_image_moves_file_and_swaps_manifest_key(tmp_path):
    project_dir, manifest = _make_project(tmp_path)
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"

    new_key = relabel_image(manifest, project_dir, "good/good0.png", "bad")

    assert new_key == "bad/good0.png"
    assert new_key in manifest["images"]
    assert "good/good0.png" not in manifest["images"]
    assert manifest["images"][new_key] == {"class": "bad", "split": "train"}

    assert not (good_dir / "good0.png").exists()
    assert (bad_dir / "good0.png").exists()

    # persisted to disk, not just in-memory
    reloaded = load_manifest(project_dir)
    assert new_key in reloaded["images"]
    assert "good/good0.png" not in reloaded["images"]


def test_relabel_image_preserves_split(tmp_path):
    project_dir, manifest = _make_project(tmp_path, ratios=(0.0, 1.0, 0.0))
    new_key = relabel_image(manifest, project_dir, "good/good0.png", "bad")
    assert manifest["images"][new_key]["split"] == "val"


def test_relabel_image_rejects_invalid_new_class(tmp_path):
    project_dir, manifest = _make_project(tmp_path)
    with pytest.raises(ValueError):
        relabel_image(manifest, project_dir, "good/good0.png", "ugly")


def test_relabel_image_rejects_same_class(tmp_path):
    project_dir, manifest = _make_project(tmp_path)
    with pytest.raises(ValueError):
        relabel_image(manifest, project_dir, "good/good0.png", "good")


def test_relabel_image_refuses_to_touch_test_split(tmp_path):
    project_dir, manifest = _make_project(tmp_path, ratios=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        relabel_image(manifest, project_dir, "good/good0.png", "bad")


def test_relabel_image_refuses_unknown_key(tmp_path):
    project_dir, manifest = _make_project(tmp_path)
    with pytest.raises(KeyError):
        relabel_image(manifest, project_dir, "good/does_not_exist.png", "bad")


def test_relabel_image_refuses_filename_collision(tmp_path):
    project_dir, manifest = _make_project(tmp_path)
    bad_dir = tmp_path / "bad_demo"
    # a file that happens to share good0.png's name is already filed under bad/
    Image.new("RGBA", (64, 64), (0, 0, 0, 255)).save(bad_dir / "good0.png")

    with pytest.raises(FileExistsError):
        relabel_image(manifest, project_dir, "good/good0.png", "bad")

    # nothing should have moved or been deleted from the manifest on failure
    assert (tmp_path / "good" / "good0.png").exists()
    assert "good/good0.png" in manifest["images"]
