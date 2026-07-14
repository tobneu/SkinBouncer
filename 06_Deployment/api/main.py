from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from minecraft_skin_downloader import MinecraftSkinDownloader

from skinbouncer_core import load_model, load_skin

app = FastAPI(title="SkinBouncer API")

model = load_model("./model/best_model.keras")
model_threshold = 0.652

class PlayerCheckRequest(BaseModel):
    player_name: str
    player_id: str | None = None

@app.get("/")
def read_root():
    return {"message": "This is the minecraft skin safety gateway"}

# TODO: This is not a secure endpoint, just for demonstration and local use only :)
@app.post("/check/player/")
def check_player(request: PlayerCheckRequest):
    if request.player_name == "" and request.player_id == "":
        raise HTTPException(status_code=400)

    dl = MinecraftSkinDownloader()
    res = {
        "player_name": request.player_name,
        "score": 0.0,
        "risk": False
    }

    if request.player_id:
        res["player_id"] = request.player_id
        img_path = f"./data/skins/id/{request.player_id}.png"
        dl.download_by_uuid(request.player_id, img_path)
    else:
        img_path = f"./data/skins/name/{request.player_name}.png"
        dl.download_by_name(request.player_name, img_path)

    loaded_skin = load_skin(img_path)
    y = float(model.predict(loaded_skin)[0][0])

    res["score"] = y
    res["risk"] = y > model_threshold

    print(f"Score for {request.player_name}: {y:.4f} | Risk: {res['risk']}")
    return res

