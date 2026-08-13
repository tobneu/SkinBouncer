"""Generates a small local sample dataset for onboarding/demoing the SkinBouncer pipeline.

This dataset is NOT committed to the repo (see .gitignore) - regenerate it by running
this script. It exists purely so a fresh clone can run the pipeline end-to-end without
needing the removed scraper or any manually-supplied data.

It is a pipeline demo, not an accuracy demo. Measured on 150 images per class over four
seeds, the CNN reaches a validation AUC of about 0.6 (observed range 0.57-0.72) - above
chance, far below the 0.983 the same architecture reaches on the real ~10,000-image
dataset it was tuned for, and unstable enough that two runs of this script can disagree
by 0.15.

The limit is a small marker on a small dataset. The network ends in
GlobalAveragePooling2D over an 8x8 feature map, so after three pooling stages an 8x8
marker has been reduced to a single cell with no shape left, and 106 training images per
class is not enough to learn what remains. For contrast, replacing the marker with a
faint tint on every pixel - a signal pooling cannot destroy - takes the same architecture
on the same split to 0.99.

Expect the pipeline to run, the metrics to be real, and the model to be mediocre. Train
on your own data for a detector that actually works.

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

Re-running is safe and picks up where it left off: images already in the output folders
count towards the target, so an interrupted run (or a Mojang rate-limit) only costs the
images it had not fetched yet.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --images-per-class 60
"""

import argparse
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

IMAGES_PER_CLASS = 150
ATTEMPTS_PER_IMAGE = 10  # ~77% hit rate observed, generous margin
REQUEST_DELAY_SECONDS = 0.15  # be polite to the Mojang API

# UV boxes the marker is stamped into, as (x, y, w, h) - head and torso, front and back.
# Coordinates match HEAD/BODY in 02_DataUnderstanding/skin.py:58-73, the only place in
# the repo defining this mapping.
#
# Four regions rather than one, because a real prohibited skin differs over the whole
# texture rather than in a single patch, and a marker on head and torso stays visible from
# any angle in the 3D preview. It does not measurably help the model: run-to-run spread on
# this dataset size is wider than the difference between one box and four.
MARKER_BOXES = [
    (8, 8, 8, 8),    # head, front
    (24, 8, 8, 8),   # head, back
    (20, 20, 8, 12),  # torso, front
    (32, 20, 8, 12),  # torso, back
]

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


def fetch_unique_skins(downloader, count, exclude_names, target_dir, on_download=None):
    """Fill target_dir up to `count` images, counting whatever is already in it.

    Existing files are the resume mechanism: a run cut short by a rate limit or a
    Ctrl-C keeps everything it fetched, and the next run only pays for the remainder.
    Their names are also what keeps the two pools disjoint across runs, since the file
    name is the account the skin came from.

    on_download runs on each newly written file before it is counted, so an image is
    only ever in the folder in its finished form.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.stem for p in target_dir.glob("*.png"))
    tried = set(exclude_names) | set(existing)
    collected = list(existing)
    if existing:
        print(f"  resuming: {len(existing)} already in {target_dir.name}/")

    max_attempts = max(count - len(collected), 0) * ATTEMPTS_PER_IMAGE
    attempts = 0
    while len(collected) < count and attempts < max_attempts:
        attempts += 1
        name = random_username()
        if name in tried:
            continue
        tried.add(name)
        out_path = target_dir / f"{name}.png"
        if downloader.download_by_name(name, str(out_path)):
            if on_download is not None:
                on_download(out_path)
            collected.append(name)
        time.sleep(REQUEST_DELAY_SECONDS)
        if attempts % 25 == 0:
            print(f"  ...{attempts} attempts, {len(collected)}/{count} collected")
    if len(collected) < count:
        raise RuntimeError(
            f"Only found {len(collected)}/{count} valid skins after {attempts} attempts. "
            "Random usernames are hit-or-miss against the real Mojang API - rerun to "
            "continue from here, the images already fetched are kept."
        )
    return collected, tried


def draw_smiley(image_path):
    """Stamp the marker into every MARKER_BOXES region. One color per image, so the
    marker is a consistent thing on a skin rather than four unrelated blobs."""
    img = Image.open(image_path).convert("RGBA")
    face_color = (*random.choice(FACE_PALETTE), 255)
    for box_x, box_y, box_w, box_h in MARKER_BOXES:
        left = box_x + (box_w - 8) // 2
        top = box_y + (box_h - 8) // 2
        for row, line in enumerate(SMILEY_PATTERN):
            for col, ch in enumerate(line):
                if ch == ".":
                    continue
                color = EYE_COLOR if ch == "B" else face_color
                img.putpixel((left + col, top + row), color)
    img.save(image_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=IMAGES_PER_CLASS,
        help=f"images to end up with in each class (default: {IMAGES_PER_CLASS}). "
             "Fewer runs faster but see this script's docstring - 50 did not generalize.",
    )
    args = parser.parse_args()

    print(f"Generating sample dataset in {OUTPUT_ROOT} (gitignored, not committed)")
    downloader = MinecraftSkinDownloader()

    print(f"Fetching {args.images_per_class} skins for good/ ...")
    good_names, used_names = fetch_unique_skins(
        downloader, args.images_per_class, set(), GOOD_DIR
    )

    print(f"Fetching {args.images_per_class} more (disjoint) skins for bad_demo/ ...")
    # Stamped as each one lands rather than in a pass at the end: an unstamped image in
    # bad_demo/ is indistinguishable from a good/ one, so an interrupted run must never
    # be able to leave one behind for a later resume to count as done.
    bad_names, _ = fetch_unique_skins(
        downloader, args.images_per_class, used_names, BAD_DEMO_DIR, on_download=draw_smiley
    )

    print(f"Done: {len(good_names)} images in good/, {len(bad_names)} images in bad_demo/")


if __name__ == "__main__":
    main()
