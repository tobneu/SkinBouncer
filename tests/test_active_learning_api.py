from pathlib import Path

from labeling_tool.api import ActiveLearningAPI


class _StubSession:
    """Duck-typed stand-in for ActiveLearningSession, so ActiveLearningAPI's
    get_state()/decide() field shapes get fast, TF-free coverage without needing a
    real trained model."""

    def __init__(self, items):
        self.items = items
        self.index = 0
        self.decisions = []

    def total(self):
        return len(self.items)

    def remaining(self):
        return len(self.items) - self.index

    def is_done(self):
        return self.index >= len(self.items)

    def current_item(self):
        return None if self.is_done() else self.items[self.index]

    def current_path(self):
        item = self.current_item()
        return item["path"] if item else None

    def decide(self, action):
        self.decisions.append(action)
        self.index += 1


def _fake_png(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"fake-png")
    return path


def test_get_state_includes_ranking_metadata(tmp_path):
    path = _fake_png(tmp_path, "a.png")
    items = [{
        "key": "bad/a.png",
        "path": path,
        "recorded_class": "good",
        "prob": 0.87,
        "suspicion": 0.87,
        "reason": "model disagrees",
    }]
    api = ActiveLearningAPI(_StubSession(items))

    state = api.get_state()

    assert state["done"] is False
    assert state["filename"] == "a.png"
    assert state["recorded_class"] == "good"
    assert state["predicted_prob"] == 0.87
    assert state["reason"] == "model disagrees"


def test_get_state_done_has_no_ranking_metadata(tmp_path):
    api = ActiveLearningAPI(_StubSession([]))
    state = api.get_state()
    assert state["done"] is True
    assert "recorded_class" not in state
    assert "predicted_prob" not in state
    assert "reason" not in state


def test_decide_forwards_action_and_returns_fresh_state(tmp_path):
    items = [
        {"key": "good/a.png", "path": _fake_png(tmp_path, "a.png"), "recorded_class": "good",
         "prob": 0.2, "suspicion": 0.2, "reason": "confident agreement"},
        {"key": "bad/b.png", "path": _fake_png(tmp_path, "b.png"), "recorded_class": "bad",
         "prob": 0.9, "suspicion": 0.1, "reason": "confident agreement"},
    ]
    session = _StubSession(items)
    api = ActiveLearningAPI(session)

    state = api.decide("bad")

    assert session.decisions == ["bad"]
    assert state["filename"] == "b.png"
    assert state["recorded_class"] == "bad"
