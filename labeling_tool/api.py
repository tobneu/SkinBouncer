"""The js_api adapter bound into the pywebview window - the only bridge between the
JS UI and Python. Kept intentionally thin: all real logic lives in ReviewSession.
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
        self._session.decide(action)
        return self.get_state()
