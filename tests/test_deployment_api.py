"""Exercises the FastAPI app in 06_Deployment/api/main.py the same way it's run in
the deployment image: main.py imported with 06_Deployment/api on sys.path (so its
bare `import minecraft_skin_downloader` resolves), and its module-level `detectors`
dict built from whatever `./models/detectors` looks like relative to the process cwd
at import time. Network calls to Mojang are stubbed out so these run offline.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from skinbouncer_core import setup_detector_project
from skinbouncer_core.train import train_detector

API_DIR = Path(__file__).resolve().parents[1] / "06_Deployment" / "api"
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


@pytest.fixture
def client(monkeypatch, tmp_path):
    # No ./models/detectors under here, so this mirrors a build with zero exported
    # detectors (build.sh's default state for a fresh clone).
    monkeypatch.chdir(tmp_path)
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


def _make_fixture_images(folder, prefix, n, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGBA", (64, 64), color).save(folder / f"{prefix}{i}.png")


def test_check_player_returns_scores_for_a_configured_detector(monkeypatch, tmp_path):
    # Same layout load_detectors() expects: ./models/detectors/<category>/{model.keras,
    # threshold.json}, built via the same train_detector() the labeling tool uses.
    monkeypatch.chdir(tmp_path)
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
