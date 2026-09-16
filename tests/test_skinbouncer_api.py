from labeling_tool.api import SkinBouncerAPI


class _StubWindow:
    """Duck-typed stand-in for pywebview's Window, so pick_folder() gets coverage
    without a real Qt event loop or native dialog."""

    def __init__(self, path):
        self._path = path
        self.calls = []

    def create_file_dialog(self, dialog_type):
        self.calls.append(dialog_type)
        return (self._path,) if self._path is not None else None


class _StubSession:
    """Same duck-typed stand-in as tests/test_active_learning_api.py's _StubSession,
    just enough for ActiveLearningAPI.get_state() to succeed without a real trained
    model - SkinBouncerAPI._open() constructs a real ActiveLearningSession, so tests
    that reach it monkeypatch that name with this class instead."""

    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.index = 0
        self.training_progress = {"status": "idle"}
        self.run_comparison = None
        self.confusion_matrix = None
        self.test_curation = {"reviewed": 0, "total": 0, "complete": False}

    def total(self):
        return 0

    def remaining(self):
        return 0

    def is_done(self):
        return True

    def current_item(self):
        return None

    def current_path(self):
        return None


def _make_project_dir(tmp_path, name, trained):
    project_dir = tmp_path / name
    project_dir.mkdir()
    (project_dir / "split_manifest.json").write_text("{}")
    if trained:
        (project_dir / "model.keras").write_text("fake")
        (project_dir / "threshold.json").write_text("{}")
    return project_dir


def _make_api(tmp_path, projects_root=None):
    return SkinBouncerAPI(projects_root or tmp_path, settings_path=tmp_path / "settings.json")


def test_get_state_starts_on_overview(tmp_path):
    api = _make_api(tmp_path)
    state = api.get_state()
    assert state == {"screen": "overview", "projects": []}


def test_get_state_lists_projects_on_disk(tmp_path):
    _make_project_dir(tmp_path, "bad_demo", trained=True)
    api = _make_api(tmp_path)
    state = api.get_state()
    assert len(state["projects"]) == 1
    assert state["projects"][0]["project_dir"] == str(tmp_path / "bad_demo")
    assert state["projects"][0]["trained"] is True


def test_pick_folder_returns_the_chosen_path(tmp_path):
    api = _make_api(tmp_path)
    api.window = _StubWindow("/some/folder")
    assert api.pick_folder() == {"path": "/some/folder"}


def test_pick_folder_returns_none_when_cancelled(tmp_path):
    api = _make_api(tmp_path)
    api.window = _StubWindow(None)
    assert api.pick_folder() == {"path": None}


def test_create_project_delegates_to_the_overview_session(tmp_path):
    api = _make_api(tmp_path)
    calls = []
    api._overview.create_project = lambda name, good, bad: calls.append((name, good, bad))

    result = api.create_project("Demo", "/good", "/bad")

    assert result == {"status": "started"}
    assert calls == [("Demo", "/good", "/bad")]


def test_create_project_returns_error_without_starting_training(tmp_path):
    api = _make_api(tmp_path)

    def _boom(name, good, bad):
        raise ValueError("no images found")

    api._overview.create_project = _boom

    result = api.create_project("Demo", "/good", "/bad")

    assert result == {"status": "error", "message": "no images found"}


def test_open_project_on_trained_project_goes_straight_to_review(tmp_path, monkeypatch):
    project_dir = _make_project_dir(tmp_path, "bad_demo", trained=True)
    monkeypatch.setattr("labeling_tool.api.ActiveLearningSession", _StubSession)
    api = _make_api(tmp_path)

    result = api.open_project(str(project_dir))

    assert result == {"status": "ok"}
    assert api.get_state()["screen"] == "review"


def test_open_project_on_untrained_project_starts_training(tmp_path):
    project_dir = _make_project_dir(tmp_path, "bad_demo", trained=False)
    api = _make_api(tmp_path)
    calls = []
    api._overview.train_existing = lambda pd: calls.append(pd)

    result = api.open_project(str(project_dir))

    assert result == {"status": "started"}
    assert calls == [project_dir]
    # still on the overview - nothing's open yet until training finishes
    assert api.get_state()["screen"] == "overview"


def test_close_project_returns_to_overview(tmp_path, monkeypatch):
    project_dir = _make_project_dir(tmp_path, "bad_demo", trained=True)
    monkeypatch.setattr("labeling_tool.api.ActiveLearningSession", _StubSession)
    api = _make_api(tmp_path)
    api.open_project(str(project_dir))
    assert api.get_state()["screen"] == "review"

    state = api.close_project()

    assert state["screen"] == "overview"


def test_get_training_progress_auto_opens_the_project_once_done(tmp_path, monkeypatch):
    project_dir = _make_project_dir(tmp_path, "bad_demo", trained=False)
    monkeypatch.setattr("labeling_tool.api.ActiveLearningSession", _StubSession)
    api = _make_api(tmp_path)
    api._overview.training_progress = {"status": "done"}
    api._overview.trained_project_dir = project_dir

    api.get_training_progress()

    assert api.get_state()["screen"] == "review"
    # consumed - a later close_project()/reopen cycle can't accidentally re-trigger it
    assert api._overview.trained_project_dir is None


def test_decide_retrain_export_delegate_to_the_open_project(tmp_path, monkeypatch):
    project_dir = _make_project_dir(tmp_path, "bad_demo", trained=True)
    monkeypatch.setattr("labeling_tool.api.ActiveLearningSession", _StubSession)
    api = _make_api(tmp_path)
    api.open_project(str(project_dir))

    calls = []
    api._project_api.decide = lambda action: calls.append(("decide", action)) or {"ok": True}
    api._project_api.retrain = lambda: calls.append(("retrain",)) or {"status": "started"}
    api._project_api.export_detector = lambda: calls.append(("export",)) or {"ok": True}

    assert api.decide("good") == {"ok": True}
    assert api.retrain() == {"status": "started"}
    assert api.export_detector() == {"ok": True}
    assert calls == [("decide", "good"), ("retrain",), ("export",)]
