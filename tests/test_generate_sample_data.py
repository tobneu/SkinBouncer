"""Covers the resume behavior of scripts/generate_sample_data.py, which is the only path
a fresh clone has to any data at all. It talks to the live Mojang API, so a run can be cut
short by a rate limit at any point - what matters is that the next run continues from
there instead of starting over, and that a half-finished run can't leave the dataset in a
state a later run mistakes for finished.

The script is loaded by path (scripts/ is CLI wrappers, not an importable package) and
handed a fake downloader, so nothing here touches the network.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_sample_data.py"

BASE_COLOR = (10, 10, 10, 255)


@pytest.fixture(scope="module")
def gsd():
    spec = importlib.util.spec_from_file_location("generate_sample_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_sample_data"] = module
    spec.loader.exec_module(module)
    return module


class FakeDownloader:
    """Writes a flat-colored skin. Every third name fails to resolve, mirroring the real
    hit rate of guessed usernames closely enough to exercise the attempt budget."""

    def __init__(self):
        self.calls = 0

    def download_by_name(self, player_name, output_path):
        self.calls += 1
        if self.calls % 3 == 0:
            return False
        Image.new("RGBA", (64, 64), BASE_COLOR).save(output_path)
        return True


def _is_stamped(path, gsd):
    """Every MARKER_BOXES region has to carry the marker - a partially stamped skin would
    still be a positive example that looks like a negative one from most angles."""
    img = Image.open(path).convert("RGBA")
    return all(
        any(
            img.getpixel((x + dx, y + dy)) != BASE_COLOR
            for dx in range(w)
            for dy in range(h)
        )
        for x, y, w, h in gsd.MARKER_BOXES
    )


@pytest.fixture(autouse=True)
def no_sleep(gsd, monkeypatch):
    monkeypatch.setattr(gsd, "REQUEST_DELAY_SECONDS", 0)


def test_fetch_unique_skins_reaches_the_requested_count(gsd, tmp_path):
    target = tmp_path / "good"
    collected, _ = gsd.fetch_unique_skins(FakeDownloader(), 5, set(), target)

    assert len(collected) == 5
    assert len(list(target.glob("*.png"))) == 5


def test_fetch_unique_skins_counts_existing_images_instead_of_refetching(gsd, tmp_path):
    target = tmp_path / "good"
    gsd.fetch_unique_skins(FakeDownloader(), 5, set(), target)

    resumed = FakeDownloader()
    collected, _ = gsd.fetch_unique_skins(resumed, 5, set(), target)

    assert len(collected) == 5
    assert len(list(target.glob("*.png"))) == 5
    # Already complete, so the resume must not have cost a single request.
    assert resumed.calls == 0


def test_fetch_unique_skins_only_fetches_the_remainder_after_an_interrupted_run(gsd, tmp_path):
    target = tmp_path / "good"
    gsd.fetch_unique_skins(FakeDownloader(), 5, set(), target)
    for path in sorted(target.glob("*.png"))[:2]:
        path.unlink()

    resumed = FakeDownloader()
    collected, _ = gsd.fetch_unique_skins(resumed, 5, set(), target)

    assert len(collected) == 5
    assert len(list(target.glob("*.png"))) == 5
    # Two images short, so at most the per-image attempt budget for two of them.
    assert 0 < resumed.calls <= 2 * gsd.ATTEMPTS_PER_IMAGE


def test_resumed_bad_pool_images_are_all_stamped(gsd, tmp_path):
    """An unstamped image in bad_demo/ is pixel-identical to a good/ one, so it would be
    a mislabeled training example. Stamping on download rather than in a final pass is
    what makes that impossible to leave behind."""
    target = tmp_path / "bad_demo"
    gsd.fetch_unique_skins(FakeDownloader(), 4, set(), target, on_download=gsd.draw_smiley)
    for path in sorted(target.glob("*.png"))[:2]:
        path.unlink()
    gsd.fetch_unique_skins(FakeDownloader(), 4, set(), target, on_download=gsd.draw_smiley)

    images = list(target.glob("*.png"))
    assert len(images) == 4
    assert all(_is_stamped(path, gsd) for path in images)


def test_pools_stay_disjoint_across_runs(gsd, tmp_path):
    """The file name is the account the skin came from, so a name landing in both pools
    would put the same image on both sides of the label."""
    good, bad = tmp_path / "good", tmp_path / "bad_demo"
    good_names, used = gsd.fetch_unique_skins(FakeDownloader(), 4, set(), good)
    bad_names, _ = gsd.fetch_unique_skins(FakeDownloader(), 4, used, bad)

    assert not set(good_names) & set(bad_names)


def test_fetch_unique_skins_raises_when_the_target_is_unreachable(gsd, tmp_path):
    class AlwaysFails:
        def download_by_name(self, player_name, output_path):
            return False

    with pytest.raises(RuntimeError, match="rerun"):
        gsd.fetch_unique_skins(AlwaysFails(), 3, set(), tmp_path / "good")
