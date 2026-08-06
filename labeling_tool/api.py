"""The js_api adapters bound into the pywebview window - the only bridge between the
JS UI and Python. Kept intentionally thin: all real logic lives in the session classes
(ReviewSession / ActiveLearningSession).
"""

import base64


class LabelingAPI:
    def __init__(self, session):
        self._session = session

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
        # None until a retrain has completed at least once this session - the
        # comparison is only meaningful once there's a "current" round to report on.
        state["run_comparison"] = self._session.run_comparison
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
