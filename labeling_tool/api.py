"""The js_api adapters bound into the pywebview window - the only bridge between the
JS UI and Python. Kept intentionally thin: all real logic lives in the session classes
(ReviewSession / ActiveLearningSession).
"""

import base64
from pathlib import Path

from .active_learning_session import ActiveLearningSession
from .overview_session import ProjectOverviewSession
from .settings import DEFAULT_SETTINGS_PATH, load_theme, save_theme


class LabelingAPI:
    def __init__(self, session, settings_path=DEFAULT_SETTINGS_PATH):
        self._session = session
        self._settings_path = settings_path

    def get_state(self):
        path = self._session.current_path()
        if path is None:
            return {
                "done": True,
                "index": self._session.index,
                "total": self._session.total(),
                "remaining": 0,
                "filename": None,
                "image_data_uri": None,
            }

        image_data_uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "done": False,
            "index": self._session.index,
            "total": self._session.total(),
            "remaining": self._session.remaining(),
            "filename": path.name,
            "image_data_uri": image_data_uri,
        }

    def decide(self, action):
        if not self._session.is_done():
            self._session.decide(action)
        return self.get_state()

    def get_settings(self):
        """Kept separate from get_state(): theme is a cross-cutting GUI setting, not
        review-session state, the same reasoning that keeps get_training_progress()
        off get_state() in ActiveLearningAPI."""
        return {"theme": load_theme(self._settings_path)}

    def set_theme(self, theme):
        save_theme(theme, self._settings_path)
        return {"status": "ok"}


class ActiveLearningAPI(LabelingAPI):
    """Same js_api contract as LabelingAPI (get_state/decide), extended with the
    ranking metadata (recorded_class/predicted_prob/reason) an ActiveLearningSession
    tracks per item, so the UI can show the user why an image was surfaced."""

    def get_state(self):
        state = super().get_state()
        # Set unconditionally (unlike the fields below) so the frontend can tell
        # whether to show the Retrain button even on the "done" screen, where
        # current_item() is None and the ranking-only fields aren't available.
        state["can_retrain"] = True
        # Same unconditional treatment as can_retrain, for the same reason: exporting
        # stays available on the "done" screen, which is where an operator who just
        # finished a review round is most likely to want it.
        state["can_export"] = True
        # None until a retrain has completed at least once this session - the
        # comparison is only meaningful once there's a "current" round to report on.
        state["run_comparison"] = self._session.run_comparison
        # Unlike run_comparison, populated from the very first launch (computed once
        # in ActiveLearningSession.__init__ against whatever checkpoint already
        # exists), so this panel is visible even before any retrain happens.
        state["confusion_matrix"] = self._session.confusion_matrix
        # How much of the frozen test split has been blind-reviewed - shown next to the
        # confusion matrix so the operator can tell how much to trust those numbers
        # before exporting.
        state["test_curation"] = self._session.test_curation
        if not state["done"]:
            item = self._session.current_item()
            state.update({
                "recorded_class": item["recorded_class"],
                "predicted_prob": item["prob"],
                "reason": item["reason"],
            })
        return state

    def retrain(self):
        # Starts training on a background thread and returns immediately - the
        # frontend polls get_training_progress() instead of waiting on this call.
        self._session.retrain()
        return {"status": "started"}

    def get_training_progress(self):
        return self._session.training_progress

    def export_detector(self):
        # Not named export() like the session method it forwards to: this one becomes a
        # property on the JS-side api object, and `export` is a reserved word there.
        #
        # Fast enough to stay a blocking call (two file copies) - unlike retrain(),
        # there's nothing here worth a background thread and a polling loop.
        return self._session.export()


class BlindTestReviewAPI(LabelingAPI):
    """Same js_api contract as LabelingAPI, but deliberately never exposes any
    model-derived field (no predicted_prob/reason/can_retrain) - this mode shows no
    model information at all, only the image's current recorded label."""

    def get_state(self):
        state = super().get_state()
        state["can_skip"] = False
        if not state["done"]:
            state["recorded_class"] = self._session.current_item()["recorded_class"]
        return state


class SkinBouncerAPI:
    """js_api for the single long-lived project-overview window (scripts/run_skinbouncer.py).
    Wraps a ProjectOverviewSession plus, once a project is open, an ActiveLearningAPI it
    delegates review/retrain/export calls to - get_state()'s "screen" field is what tells
    the frontend which of the two modes it's currently looking at.

    Not a LabelingAPI subclass: LabelingAPI is built around one fixed session handed in
    at construction, whereas this API's "current project" changes over the window's
    lifetime as projects are opened and closed.
    """

    def __init__(self, projects_root, settings_path=DEFAULT_SETTINGS_PATH):
        self._overview = ProjectOverviewSession(projects_root)
        self._project_api = None
        self._settings_path = settings_path
        # Set by the entrypoint script right after webview.create_window() - js_api is
        # bound at construction time, but nothing stops attaching the Window afterward.
        self.window = None

    def get_settings(self):
        return {"theme": load_theme(self._settings_path)}

    def set_theme(self, theme):
        save_theme(theme, self._settings_path)
        return {"status": "ok"}

    def _open(self, project_dir):
        session = ActiveLearningSession(project_dir)
        self._project_api = ActiveLearningAPI(session, settings_path=self._settings_path)

    def _maybe_finish_training(self):
        """Cold-start training (wizard or auto-train-on-open) has no session to attach
        to until it succeeds - this is where that session finally gets constructed,
        the moment a caller next asks for state or progress after training finished.

        trained_project_dir is cleared unconditionally, before _open() runs: a project
        that trains fine but fails to open (e.g. a stale manifest pointing at images
        that no longer exist) must not be retried on every subsequent poll forever -
        it failed once, it's reported once, same as any other error here."""
        if self._project_api is None and self._overview.trained_project_dir is not None:
            project_dir = self._overview.trained_project_dir
            self._overview.trained_project_dir = None
            try:
                self._open(project_dir)
            except Exception as e:
                self._overview.training_progress = {"status": "error", "error": str(e)}

    def get_state(self):
        self._maybe_finish_training()
        if self._project_api is not None:
            state = self._project_api.get_state()
            state["screen"] = "review"
            return state
        return {"screen": "overview", "projects": self._overview.list_projects()}

    def get_training_progress(self):
        self._maybe_finish_training()
        if self._project_api is not None:
            return self._project_api.get_training_progress()
        return self._overview.training_progress

    def pick_folder(self):
        # Local import: importing this module must not require the labeling-tool
        # extra (pywebview) to be installed - only actually reaching this method
        # (from inside a real GUI session) should. See labeling_tool/app.py and
        # friends for the same lazy-import reasoning applied to whole modules.
        import webview

        result = self.window.create_file_dialog(webview.FileDialog.FOLDER)
        return {"path": result[0] if result else None}

    def create_project(self, name, good_dir, bad_dir):
        try:
            self._overview.create_project(name, good_dir, bad_dir)
        except Exception as e:
            return {"status": "error", "message": str(e)}
        return {"status": "started"}

    def open_project(self, project_dir):
        project_dir = Path(project_dir)
        if (project_dir / "model.keras").exists() and (project_dir / "threshold.json").exists():
            try:
                self._open(project_dir)
            except Exception as e:
                return {"status": "error", "message": str(e)}
            return {"status": "ok"}
        try:
            self._overview.train_existing(project_dir)
        except Exception as e:
            return {"status": "error", "message": str(e)}
        return {"status": "started"}

    def close_project(self):
        self._project_api = None
        return self.get_state()

    def decide(self, action):
        return self._project_api.decide(action)

    def retrain(self):
        return self._project_api.retrain()

    def export_detector(self):
        return self._project_api.export_detector()
