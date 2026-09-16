"""Exercises the FastAPI app in api/main.py the same way it's run in
the deployment image: main.py imported with api on sys.path (so its
bare `import minecraft_skin_downloader` resolves), and its module-level `detectors`
dict built at import time from the folder SKINBOUNCER_DETECTORS_DIR points at.

Setting that env var is what keeps these tests off the repo's own
api/models/detectors - main.py otherwise resolves the folder relative
to its own location, which in a source checkout is the maintainer's real exports.
Network calls to Mojang are stubbed out so these run offline.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from skinbouncer_core import export_detector, setup_detector_project
from skinbouncer_core.train import train_detector

API_DIR = Path(__file__).resolve().parents[1] / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))


def _import_main(module_name):
    """Fresh import of main.py under its own module name, so tests that need
    different `detectors` states (e.g. one with a trained detector configured)
    don't share the module-level `detectors` dict computed at import time."""
    spec = importlib.util.spec_from_file_location(module_name, API_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _print_response(label, res):
    # Only visible with `pytest -s` (pytest captures stdout by default).
    print(f"\n{label}: {res.status_code}\n{json.dumps(res.json(), indent=2)}")


def _fake_download(output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64)).save(output_path)
    return True


def _stub_downloader(monkeypatch, module):
    monkeypatch.setattr(
        module.MinecraftSkinDownloader,
        "download_by_name",
        lambda self, player_name, output_path: _fake_download(output_path),
    )
    monkeypatch.setattr(
        module.MinecraftSkinDownloader,
        "download_by_uuid",
        lambda self, uuid, output_path: _fake_download(output_path),
    )


def _point_at_detectors(monkeypatch, detectors_dir):
    """main.py reads SKINBOUNCER_DETECTORS_DIR at import time, so this has to happen
    before _import_main()."""
    monkeypatch.setenv("SKINBOUNCER_DETECTORS_DIR", str(detectors_dir))


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Points at a folder that doesn't exist, mirroring a build with zero exported
    # detectors (build.sh's default state for a fresh clone).
    _point_at_detectors(monkeypatch, tmp_path / "models" / "detectors")
    module = _import_main("deployment_api_main")
    _stub_downloader(monkeypatch, module)
    return TestClient(module.app)


def test_read_root_lists_detectors(client):
    res = client.get("/")
    _print_response("GET /", res)

    assert res.status_code == 200
    body = res.json()
    assert body["message"] == "This is the minecraft skin safety gateway"
    assert body["detectors"] == []


def test_check_player_by_name(client):
    res = client.post("/check/player/", json={"player_name": "SpiderMan"})
    _print_response("POST /check/player/ (by name)", res)

    assert res.status_code == 200
    assert res.json() == {"player_name": "SpiderMan", "categories": {}}


def test_check_player_by_id(client):
    player_id = "069a79f4-44e9-4726-a5be-fca90e38aaf5"
    res = client.post(
        "/check/player/",
        json={"player_name": "SpiderMan", "player_id": player_id},
    )
    _print_response("POST /check/player/ (by id)", res)

    assert res.status_code == 200
    body = res.json()
    assert body["player_id"] == player_id
    assert body["categories"] == {}


def test_check_player_rejects_empty_identifiers(client):
    res = client.post("/check/player/", json={"player_name": "", "player_id": ""})
    _print_response("POST /check/player/ (empty identifiers)", res)

    assert res.status_code == 400


def test_check_player_requires_player_name_field(client):
    res = client.post("/check/player/", json={})
    _print_response("POST /check/player/ (missing player_name)", res)

    assert res.status_code == 422


def test_check_player_rejects_empty_name_without_an_id(client):
    """player_id defaults to None rather than "", so an empty name on its own has to be
    caught by the same guard - otherwise it falls through into a Mojang lookup for the
    empty string."""
    res = client.post("/check/player/", json={"player_name": ""})
    _print_response("POST /check/player/ (empty name, no id)", res)

    assert res.status_code == 400


def test_check_player_returns_404_when_the_skin_cannot_be_fetched(monkeypatch, tmp_path):
    """The downloader reports every failure - unknown player, default Steve/Alex skin,
    Mojang unreachable - by returning False without writing the file. Scoring the file
    anyway turns all of them into a 500 traceback."""
    _point_at_detectors(monkeypatch, tmp_path / "models" / "detectors")
    module = _import_main("deployment_api_main_download_fails")
    monkeypatch.setattr(
        module.MinecraftSkinDownloader,
        "download_by_name",
        lambda self, player_name, output_path: False,
    )
    client = TestClient(module.app)

    res = client.post("/check/player/", json={"player_name": "NoSuchPlayer"})
    _print_response("POST /check/player/ (download fails)", res)

    assert res.status_code == 404
    assert "NoSuchPlayer" in res.json()["detail"]


def test_check_player_does_not_keep_the_downloaded_skin(monkeypatch, tmp_path):
    """Every checked skin used to be written to a cwd-relative folder keyed by the
    request's player_name and left there, so a long-lived server grew without bound and
    a name like "../../x" escaped the folder."""
    _point_at_detectors(monkeypatch, tmp_path / "models" / "detectors")
    module = _import_main("deployment_api_main_no_leftovers")
    written = []

    def record_and_download(self, player_name, output_path):
        written.append(Path(output_path))
        return _fake_download(output_path)

    monkeypatch.setattr(module.MinecraftSkinDownloader, "download_by_name", record_and_download)
    client = TestClient(module.app)

    res = client.post("/check/player/", json={"player_name": "../../escaped"})

    assert res.status_code == 200
    assert len(written) == 1
    assert not written[0].exists()
    assert not written[0].parent.exists()


def _make_fixture_images(folder, prefix, n, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGBA", (64, 64), color).save(folder / f"{prefix}{i}.png")


def test_check_player_returns_scores_for_a_configured_detector(monkeypatch, tmp_path):
    # Same layout load_detectors() expects: ./models/detectors/<category>/{model.keras,
    # threshold.json}, built via the same train_detector() the labeling tool uses.
    _point_at_detectors(monkeypatch, tmp_path / "models" / "detectors")
    good_dir = tmp_path / "_src_good"
    bad_dir = tmp_path / "_src_bad"
    _make_fixture_images(good_dir, "good", 16, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", 16, (200, 0, 0, 255))

    category_dir = tmp_path / "models" / "detectors" / "nsfw"
    setup_detector_project(good_dir, bad_dir, category_dir)
    train_detector(category_dir, epochs=1, batch_size=8)

    module = _import_main("deployment_api_main_with_detector")
    _stub_downloader(monkeypatch, module)
    client = TestClient(module.app)

    res = client.post("/check/player/", json={"player_name": "SpiderMan"})
    _print_response("POST /check/player/ (with a configured detector)", res)

    assert res.status_code == 200
    categories = res.json()["categories"]
    assert set(categories.keys()) == {"nsfw"}
    assert 0.0 <= categories["nsfw"]["score"] <= 1.0
    assert isinstance(categories["nsfw"]["risk"], bool)


def test_api_loads_and_scores_a_detector_produced_by_export_detector(monkeypatch, tmp_path):
    """Closes the loop between the two halves of the pipeline: the labeling tool's
    Export writes a folder, and load_detectors() reads one. Nothing else checks that
    those two agree on the layout, filenames and threshold format - a change to either
    side alone would otherwise only surface as a silently detector-less deployment.
    """
    _point_at_detectors(monkeypatch, tmp_path / "models" / "detectors")
    good_dir = tmp_path / "_src_good"
    bad_dir = tmp_path / "hate_spiders"
    _make_fixture_images(good_dir, "good", 16, (0, 200, 0, 255))
    _make_fixture_images(bad_dir, "bad", 16, (200, 0, 0, 255))

    # Trained somewhere entirely unrelated to the deployment layout, the way a real
    # detector project is - export is what puts it where the API can find it.
    project_dir = tmp_path / "detector_projects" / "spider_project"
    setup_detector_project(good_dir, bad_dir, project_dir)
    train_detector(project_dir, epochs=1, batch_size=8)

    result = export_detector(project_dir, detectors_dir=tmp_path / "models" / "detectors")

    module = _import_main("deployment_api_main_from_export")
    _stub_downloader(monkeypatch, module)
    client = TestClient(module.app)

    root = client.get("/")
    _print_response("GET / (detector from export_detector)", root)
    assert root.json()["detectors"] == ["hate_spiders"]

    res = client.post("/check/player/", json={"player_name": "SpiderMan"})
    _print_response("POST /check/player/ (detector from export_detector)", res)

    assert res.status_code == 200
    scored = res.json()["categories"]["hate_spiders"]
    assert 0.0 <= scored["score"] <= 1.0
    # The API's accept/reject call must use the threshold export copied over, not the
    # 0.5 default that would silently take over if threshold.json failed to load.
    assert scored["risk"] == (scored["score"] > result["threshold"])
