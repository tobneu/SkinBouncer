"""Generates a small local sample dataset for onboarding/demoing the SkinBouncer pipeline.

This dataset is NOT committed to the repo (see .gitignore) - regenerate it by running
this script. It exists purely so a fresh clone can run the pipeline end-to-end without
needing the removed scraper or any manually-supplied data. It is sized only for
pipeline smoke-testing, not for training a production-quality model.

Provenance:
- good/: real Minecraft player skins, fetched live from the official public Mojang API
  (api.mojang.com, sessionserver.mojang.com) by resolving randomly generated candidate
  usernames until enough resolve to a real, skinned account - the same technique the
  project's original Kaggle source dataset was built with (see
  02_DataUnderstanding/DataUnderstanding.ipynb). No scraping, no third-party site, no
  ToS concern - this is the same public API the kept SkinsFromUuid downloader uses.
- bad_demo/: a disjoint set of real skins (same source, no overlap with good/) with a
  synthetic, self-drawn colorful smiley stamped onto the front-torso ("belly") region as
  a stand-in for a real prohibited symbol. The smiley is drawn pixel-by-pixel with
  Pillow, not a copied asset, so there is no copyright question.

Usage:
    python scripts/generate_sample_data.py
"""

import random
import string
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "02_DataUnderstanding" / "Mining" / "SkinsFromUuid"))
from minecraft_skin_downloader import MinecraftSkinDownloader  # noqa: E402

OUTPUT_ROOT = ROOT / "sample_data"
GOOD_DIR = OUTPUT_ROOT / "good"
BAD_DEMO_DIR = OUTPUT_ROOT / "bad_demo"

IMAGES_PER_CLASS = 50
MAX_ATTEMPTS_PER_POOL = IMAGES_PER_CLASS * 10  # ~77% hit rate observed, generous margin
REQUEST_DELAY_SECONDS = 0.15  # be polite to the Mojang API

# Front-torso ("belly") UV box - same coordinates as BODY["front"] in
# 02_DataUnderstanding/skin.py:69, the only place in the repo defining this mapping.
BELLY_BOX_X, BELLY_BOX_Y, BELLY_BOX_W, BELLY_BOX_H = 20, 20, 8, 12

WORDS = [
    "shadow", "dragon", "wolf", "tiger", "storm", "blaze", "frost", "night", "star",
    "moon", "fire", "steel", "iron", "gold", "silver", "dark", "light", "king", "queen",
    "knight", "hunter", "ranger", "wizard", "mage", "archer", "phoenix", "raven", "hawk",
    "eagle", "lion", "bear", "fox", "ghost", "hero", "legend", "master", "warrior",
    "pirate", "ninja", "viking",
]

# 8x8 pixel-art smiley, blitted onto the belly box (vertically centered in its 12px
# height). '.' = transparent/skip, 'Y' = face (randomized color per image), 'B' = eyes/mouth.
SMILEY_PATTERN = [
    ".YYYYYY.",
    "YYYYYYYY",
    "YBYYYYBY",
    "YYYYYYYY",
    "YYBYYBYY",
    "YYYBBYYY",
    "YYYYYYYY",
    ".YYYYYY.",
]
FACE_PALETTE = [
    (255, 221, 0), (255, 105, 180), (0, 206, 209),
    (50, 205, 50), (255, 140, 0), (186, 85, 211),
]
EYE_COLOR = (20, 20, 20, 255)


def random_username():
    word = random.choice(WORDS)
    suffix = "".join(random.choices(string.digits, k=random.choice([0, 1, 2, 3])))
    return (word + suffix)[:16]


def fetch_unique_skins(downloader, count, exclude_names, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    collected = []
    tried = set(exclude_names)
    attempts = 0
    while len(collected) < count and attempts < MAX_ATTEMPTS_PER_POOL:
        attempts += 1
        name = random_username()
        if name in tried:
            continue
        tried.add(name)
        out_path = target_dir / f"{name}.png"
        if downloader.download_by_name(name, str(out_path)):
            collected.append(name)
        time.sleep(REQUEST_DELAY_SECONDS)
        if attempts % 25 == 0:
            print(f"  ...{attempts} attempts, {len(collected)}/{count} collected")
    if len(collected) < count:
        raise RuntimeError(
            f"Only found {len(collected)}/{count} valid skins after {attempts} attempts. "
            "Random usernames are hit-or-miss against the real Mojang API - just rerun."
        )
    return collected, tried


def draw_smiley(image_path):
    img = Image.open(image_path).convert("RGBA")
    face_color = (*random.choice(FACE_PALETTE), 255)
    top = BELLY_BOX_Y + (BELLY_BOX_H - 8) // 2
    for row, line in enumerate(SMILEY_PATTERN):
        for col, ch in enumerate(line):
            if ch == ".":
                continue
            color = EYE_COLOR if ch == "B" else face_color
            img.putpixel((BELLY_BOX_X + col, top + row), color)
    img.save(image_path)


def main():
    print(f"Generating sample dataset in {OUTPUT_ROOT} (gitignored, not committed)")
    downloader = MinecraftSkinDownloader()

    print(f"Fetching {IMAGES_PER_CLASS} skins for good/ ...")
    good_names, used_names = fetch_unique_skins(downloader, IMAGES_PER_CLASS, set(), GOOD_DIR)

    print(f"Fetching {IMAGES_PER_CLASS} more (disjoint) skins for bad_demo/ base ...")
    bad_names, _ = fetch_unique_skins(downloader, IMAGES_PER_CLASS, used_names, BAD_DEMO_DIR)

    print("Stamping synthetic smiley marker onto bad_demo/ images...")
    for name in bad_names:
        draw_smiley(BAD_DEMO_DIR / f"{name}.png")

    print(f"Done: {len(good_names)} images in good/, {len(bad_names)} images in bad_demo/")


if __name__ == "__main__":
    main()
