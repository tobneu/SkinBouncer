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
        if not state["done"]:
            item = self._session.current_item()
            state.update({
                "recorded_class": item["recorded_class"],
                "predicted_prob": item["prob"],
                "reason": item["reason"],
            })
        return state
