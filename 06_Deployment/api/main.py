import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from minecraft_skin_downloader import MinecraftSkinDownloader

from skinbouncer_core import load_model, load_skin

DETECTORS_DIR = Path("./models/detectors")

app = FastAPI(title="SkinBouncer API")


def load_detectors(detectors_dir: Path = DETECTORS_DIR):
    """Load every detector found under detectors_dir. Each detector is a subfolder
    named after its category, containing model.keras + threshold.json. Missing or
    incomplete subfolders are skipped; a missing detectors_dir yields no detectors
    rather than an error, so the API starts fine with zero detectors configured."""
    detectors = {}
    if not detectors_dir.is_dir():
        return detectors

    for category_dir in sorted(detectors_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        model_path = category_dir / "model.keras"
        threshold_path = category_dir / "threshold.json"
        if not model_path.exists() or not threshold_path.exists():
            continue

        with open(threshold_path) as f:
            threshold = json.load(f)["threshold"]

        detectors[category_dir.name] = {
            "model": load_model(model_path),
            "threshold": threshold,
        }
        print(f"Loaded detector '{category_dir.name}' (threshold={threshold})")

    return detectors


detectors = load_detectors()


class PlayerCheckRequest(BaseModel):
    player_name: str
    player_id: str | None = None


@app.get("/")
def read_root():
    return {
        "message": "This is the minecraft skin safety gateway",
        "detectors": list(detectors.keys()),
    }


# TODO: This is not a secure endpoint, just for demonstration and local use only :)
@app.post("/check/player/")
def check_player(request: PlayerCheckRequest):
    if request.player_name == "" and request.player_id == "":
        raise HTTPException(status_code=400)

    dl = MinecraftSkinDownloader()
    res = {
        "player_name": request.player_name,
        "categories": {},
    }

    if request.player_id:
        res["player_id"] = request.player_id
        img_path = f"./data/skins/id/{request.player_id}.png"
        dl.download_by_uuid(request.player_id, img_path)
    else:
        img_path = f"./data/skins/name/{request.player_name}.png"
        dl.download_by_name(request.player_name, img_path)

    loaded_skin = load_skin(img_path)

    for category, detector in detectors.items():
        score = float(detector["model"].predict(loaded_skin, verbose=0)[0][0])
        risk = score > detector["threshold"]
        res["categories"][category] = {"score": score, "risk": risk}
        print(f"[{category}] score for {request.player_name}: {score:.4f} | risk: {risk}")

    return res
