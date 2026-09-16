import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from minecraft_skin_downloader import MinecraftSkinDownloader

from skinbouncer_core import load_model, load_skin

# Resolved relative to this file rather than the process cwd, so `uvicorn main:app`
# behaves the same from the repo root, from api/, and inside the image -
# main.py sits next to models/detectors/ in all three (see api/Dockerfile).
# The env var lets a deployment point at a detectors folder mounted elsewhere.
DETECTORS_DIR = Path(
    os.environ.get("SKINBOUNCER_DETECTORS_DIR")
    or Path(__file__).resolve().parent / "models" / "detectors"
)

app = FastAPI(title="SkinBouncer API")


def load_detectors(detectors_dir: Path = DETECTORS_DIR):
    """Load every detector found under detectors_dir. Each detector is a subfolder
    named after its category, containing model.keras + threshold.json. Missing or
    incomplete subfolders are skipped; a missing detectors_dir yields no detectors
    rather than an error, so the API starts fine with zero detectors configured."""
    detectors = {}
    if detectors_dir.is_dir():
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

    if not detectors:
        # Zero detectors is a valid state, but every /check/player/ then answers with an
        # empty "categories" dict and HTTP 200 - indistinguishable from "nothing was
        # flagged" unless the operator is told. Say so loudly at startup instead.
        print(
            f"WARNING: no detectors found under {detectors_dir}. /check/player/ will "
            f"score nothing. Export one first: python scripts/export_detector.py "
            f"--project-dir <your-project>",
            file=sys.stderr,
        )
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
    if not request.player_name and not request.player_id:
        raise HTTPException(status_code=400, detail="player_name or player_id is required")

    dl = MinecraftSkinDownloader()
    res = {
        "player_name": request.player_name,
        "categories": {},
    }

    # A temp dir keeps the request identifier out of the filesystem path entirely, and
    # bounds disk use to one skin at a time - a long-lived server would otherwise keep
    # every skin it has ever checked.
    with tempfile.TemporaryDirectory(prefix="skinbouncer-") as tmp_dir:
        img_path = Path(tmp_dir) / "skin.png"

        if request.player_id:
            res["player_id"] = request.player_id
            downloaded = dl.download_by_uuid(request.player_id, str(img_path))
        else:
            downloaded = dl.download_by_name(request.player_name, str(img_path))

        # The downloader reports failure by returning False, and every reason is a
        # normal condition rather than a server fault: unknown player, a player still
        # on the default Steve/Alex skin (no SKIN texture on the profile at all), or
        # Mojang being unreachable. Loading the file that was never written would turn
        # all of them into a 500.
        if not downloaded:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Could not fetch a skin for "
                    f"{request.player_id or request.player_name!r}. The player may not "
                    f"exist, may still use the default skin, or Mojang may be "
                    f"unreachable."
                ),
            )

        loaded_skin = load_skin(img_path)

    for category, detector in detectors.items():
        score = float(detector["model"].predict(loaded_skin, verbose=0)[0][0])
        risk = score > detector["threshold"]
        res["categories"][category] = {"score": score, "risk": risk}
        print(f"[{category}] score for {request.player_name}: {score:.4f} | risk: {risk}")

    return res
