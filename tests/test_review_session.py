from labeling_tool.review_session import ReviewSession


def _make_images(folder, names):
    for name in names:
        (folder / name).write_bytes(b"fake-png")


def test_glob_picks_up_only_png_sorted(tmp_path):
    _make_images(tmp_path, ["b.png", "a.png", "c.txt"])
    session = ReviewSession(tmp_path)
    assert [p.name for p in session.items] == ["a.png", "b.png"]


def test_decide_moves_file_to_correct_subfolder(tmp_path):
    _make_images(tmp_path, ["a.png", "b.png", "c.png"])
    session = ReviewSession(tmp_path)

    session.decide("good")
    session.decide("bad")
    session.decide("skip")

    assert (tmp_path / "good" / "a.png").exists()
    assert (tmp_path / "bad" / "b.png").exists()
    assert (tmp_path / "skip" / "c.png").exists()
    assert not (tmp_path / "a.png").exists()


def test_decide_rejects_unknown_action(tmp_path):
    _make_images(tmp_path, ["a.png"])
    session = ReviewSession(tmp_path)
    try:
        session.decide("maybe")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown action")


def test_progress_transitions_across_full_pass(tmp_path):
    _make_images(tmp_path, ["a.png", "b.png"])
    session = ReviewSession(tmp_path)

    assert session.total() == 2
    assert session.remaining() == 2
    assert not session.is_done()

    session.decide("good")
    assert session.remaining() == 1
    assert not session.is_done()

    session.decide("bad")
    assert session.remaining() == 0
    assert session.is_done()
    assert session.current_path() is None


def test_empty_folder_is_done_immediately(tmp_path):
    session = ReviewSession(tmp_path)
    assert session.total() == 0
    assert session.is_done()
    assert session.current_path() is None


def test_relaunch_on_partially_processed_folder_does_not_recount_moved_files(tmp_path):
    _make_images(tmp_path, ["a.png", "b.png", "c.png"])
    first_session = ReviewSession(tmp_path)
    first_session.decide("good")

    second_session = ReviewSession(tmp_path)
    assert second_session.total() == 2
    assert [p.name for p in second_session.items] == ["b.png", "c.png"]
