from labeling_tool.api import LabelingAPI
from labeling_tool.review_session import ReviewSession


def _make_images(folder, names):
    for name in names:
        (folder / name).write_bytes(b"fake-png")


def test_decide_after_done_does_not_raise(tmp_path):
    _make_images(tmp_path, ["a.png"])
    api = LabelingAPI(ReviewSession(tmp_path))

    state = api.decide("good")
    assert state["done"] is True

    # Simulates a duplicate call arriving after the session is already
    # exhausted (e.g. keyboard auto-repeat racing the first response).
    state = api.decide("good")
    assert state["done"] is True
    assert state["image_data_uri"] is None


def test_get_state_done_has_no_image_data(tmp_path):
    api = LabelingAPI(ReviewSession(tmp_path))
    state = api.get_state()
    assert state == {
        "done": True,
        "index": 0,
        "total": 0,
        "remaining": 0,
        "filename": None,
        "image_data_uri": None,
    }
