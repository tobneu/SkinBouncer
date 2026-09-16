import pytest
from PIL import Image

from skinbouncer_core import setup_detector_project
from labeling_tool.overview_session import ProjectOverviewSession, _slugify, _unique_project_dir


def _make_fixture_images(folder, prefix, n, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGBA", (64, 64), color).save(folder / f"{prefix}{i}.png")


def _make_source_folders(tmp_path, n=12):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad_demo"
    _make_fixture_images(good_dir, "good", n, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", n, (200, 0, 0, 255))
    return good_dir, bad_dir


# --- pure helpers: no fixtures, no TF ---

def test_slugify_lowercases_and_replaces_punctuation():
    assert _slugify("Hate Symbols!") == "hate_symbols"


def test_slugify_empty_name_falls_back_to_project():
    assert _slugify("   ") == "project"


def test_unique_project_dir_appends_suffix_on_collision(tmp_path):
    (tmp_path / "demo").mkdir()
    assert _unique_project_dir(tmp_path, "demo") == tmp_path / "demo_2"


def test_unique_project_dir_no_collision(tmp_path):
    assert _unique_project_dir(tmp_path, "demo") == tmp_path / "demo"


# --- list_projects: no training, just manifests on disk ---

def test_list_projects_empty_root(tmp_path):
    session = ProjectOverviewSession(tmp_path / "does_not_exist")
    assert session.list_projects() == []


def test_list_projects_skips_dirs_without_a_manifest(tmp_path):
    (tmp_path / "not_a_project").mkdir()
    session = ProjectOverviewSession(tmp_path)
    assert session.list_projects() == []


def test_list_projects_reports_display_name_and_trained_state(tmp_path):
    good_dir, bad_dir = _make_source_folders(tmp_path)
    setup_detector_project(good_dir, bad_dir, tmp_path / "bad_demo_project", name="Hate Symbols")
    setup_detector_project(good_dir, bad_dir, tmp_path / "untrained_project")

    session = ProjectOverviewSession(tmp_path)
    projects = {p["project_dir"]: p for p in session.list_projects()}

    trained_dir = str(tmp_path / "bad_demo_project")
    untrained_dir = str(tmp_path / "untrained_project")
    assert projects[trained_dir]["display_name"] == "Hate Symbols"
    assert projects[trained_dir]["trained"] is False  # no model.keras written yet
    assert projects[untrained_dir]["display_name"] == "bad_demo"  # falls back to category


# --- create_project / train_existing: real (tiny) training, mirroring
# tests/test_active_learning_session.py's epochs=1, batch_size=8, .join() pattern ---

def test_create_project_trains_and_reports_progress(tmp_path):
    good_dir, bad_dir = _make_source_folders(tmp_path)
    session = ProjectOverviewSession(tmp_path)

    session.create_project("Demo", good_dir, bad_dir, epochs=1, batch_size=8)
    session._thread.join()

    assert session.training_progress["status"] == "done"
    project_dir = tmp_path / "demo"
    assert session.trained_project_dir == project_dir
    assert (project_dir / "model.keras").exists()
    assert (project_dir / "threshold.json").exists()


def test_create_project_error_path_does_not_crash(tmp_path, monkeypatch):
    good_dir, bad_dir = _make_source_folders(tmp_path)
    session = ProjectOverviewSession(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr("labeling_tool.overview_session.train_detector", _boom)

    session.create_project("Demo", good_dir, bad_dir, epochs=1, batch_size=8)
    session._thread.join()

    assert session.training_progress["status"] == "error"
    assert session.training_progress["error"] == "simulated training failure"
    assert session.trained_project_dir is None
    # the manifest itself was still created (setup_detector_project runs
    # synchronously, before the background thread that failed)
    assert (tmp_path / "demo" / "split_manifest.json").exists()


def test_train_existing_trains_a_project_created_without_the_wizard(tmp_path):
    good_dir, bad_dir = _make_source_folders(tmp_path)
    project_dir = tmp_path / "cli_made"
    setup_detector_project(good_dir, bad_dir, project_dir)
    session = ProjectOverviewSession(tmp_path)

    session.train_existing(project_dir, epochs=1, batch_size=8)
    session._thread.join()

    assert session.training_progress["status"] == "done"
    assert session.trained_project_dir == project_dir
    assert (project_dir / "model.keras").exists()


def test_start_cold_start_training_refuses_concurrent_runs(tmp_path):
    good_dir, bad_dir = _make_source_folders(tmp_path)
    session = ProjectOverviewSession(tmp_path)
    session.training_progress = {"status": "running"}

    with pytest.raises(RuntimeError):
        session.create_project("Demo", good_dir, bad_dir, epochs=1, batch_size=8)
