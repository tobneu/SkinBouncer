from labeling_tool.api import BlindTestReviewAPI


class _StubSession:
    """Duck-typed stand-in for BlindTestReviewSession, so BlindTestReviewAPI's
    get_state() field shape gets fast coverage without a real project/manifest."""

    def __init__(self, items, index=0, total=None):
        self.items = items
        self.index = index
        self._total = total if total is not None else len(items)
        self._pos = 0

    def total(self):
        return self._total

    def remaining(self):
        return self._total - self.index

    def is_done(self):
        return self._pos >= len(self.items)

    def current_item(self):
        return None if self.is_done() else self.items[self._pos]

    def current_path(self):
        item = self.current_item()
        return item["path"] if item else None

    def decide(self, action):
        self._pos += 1
        self.index += 1


def _fake_png(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"fake-png")
    return path


def test_get_state_includes_recorded_class_but_no_model_fields(tmp_path):
    path = _fake_png(tmp_path, "a.png")
    items = [{"key": "good/a.png", "path": path, "recorded_class": "good"}]
    api = BlindTestReviewAPI(_StubSession(items))

    state = api.get_state()

    assert state["done"] is False
    assert state["filename"] == "a.png"
    assert state["recorded_class"] == "good"
    assert "predicted_prob" not in state
    assert "reason" not in state
    assert "can_retrain" not in state
    assert state["can_skip"] is False


def test_get_state_done_never_has_recorded_class(tmp_path):
    api = BlindTestReviewAPI(_StubSession([]))
    state = api.get_state()
    assert state["done"] is True
    assert "recorded_class" not in state
    assert state["can_skip"] is False


def test_decide_forwards_action_and_returns_fresh_state(tmp_path):
    items = [
        {"key": "good/a.png", "path": _fake_png(tmp_path, "a.png"), "recorded_class": "good"},
        {"key": "bad/b.png", "path": _fake_png(tmp_path, "b.png"), "recorded_class": "bad"},
    ]
    session = _StubSession(items)
    api = BlindTestReviewAPI(session)

    state = api.decide("good")

    assert session.index == 1
    assert state["filename"] == "b.png"
    assert state["recorded_class"] == "bad"
