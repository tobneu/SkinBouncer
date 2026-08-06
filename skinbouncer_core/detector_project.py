"""Detector project setup: stratified train/val/test split with a frozen test set.

A "detector project" is the local working state for training one binary
good-vs-bad_<category> detector. Its manifest (`split_manifest.json`, stored in the
project directory) is the single source of truth for which image belongs to which
split, and is treated as append-only for train/val and permanently frozen for test:

- On first setup, every image found in `good_dir`/`bad_dir` is stratified by class
  into train/val/test at the given ratios (default 70/15/15).
- On any later setup call (e.g. after more images were added to good_dir/bad_dir,
  or after an active-learning round moved an image between classes), only images not
  yet in the manifest are assigned a split - existing assignments are left untouched.
  New images are only ever assigned to train/val, never test, so the test set never
  grows or changes after the first run. This is what keeps test-set metrics (used to
  gate export decisions) trustworthy across any number of retrain rounds.

Manifest schema (schema_version 1):
    {
        "schema_version": 1,
        "good_dir": "<path good images were read from>",
        "bad_dir": "<path bad images were read from>",
        "category": "<bad_dir's folder name>",
        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "seed": 67,
        "images": {
            "good/<filename>": {"class": "good", "split": "train" | "val" | "test"},
            "bad/<filename>":  {"class": "bad",  "split": "train" | "val" | "test"},
            ...
        }
    }

Keys in "images" are "<class>/<filename>", not bare filenames, so a good/ and a bad/
image that happen to share a filename can't collide.
"""

import json
import os
import random
import shutil
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "split_manifest.json"
DEFAULT_RATIOS = (0.7, 0.15, 0.15)


def _write_json(path, data):
    """Write via a temp file + atomic rename, so a crash or interrupt mid-write can
    never leave a partially-written / corrupt JSON file at `path`."""
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def _list_images(folder):
    return sorted(p.name for p in Path(folder).glob("*.png"))


def _stratified_assign(filenames, ratios, seed, include_test):
    """Deterministically shuffle filenames and assign splits by ratio. If include_test
    is False, the val/test share collapses into a single train-vs-val split (used when
    adding new images to an existing project, so test never grows)."""
    train_ratio, val_ratio, test_ratio = ratios
    names = list(filenames)
    random.Random(seed).shuffle(names)
    n = len(names)

    if include_test:
        n_val = round(n * val_ratio)
        n_test = round(n * test_ratio)
        n_train = n - n_val - n_test
        assignment = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    else:
        n_val = round(n * (val_ratio / (train_ratio + val_ratio)))
        assignment = ["train"] * (n - n_val) + ["val"] * n_val

    return dict(zip(names, assignment))


def manifest_path(project_dir):
    return Path(project_dir) / MANIFEST_FILENAME


def load_manifest(project_dir):
    return json.loads(manifest_path(project_dir).read_text())


def save_manifest(manifest, project_dir):
    _write_json(manifest_path(project_dir), manifest)


def setup_detector_project(good_dir, bad_dir, project_dir, ratios=DEFAULT_RATIOS, seed=67):
    """Create (or incrementally update) a detector project's split manifest. Returns
    the manifest dict. Safe to call repeatedly - see module docstring for the
    freeze-on-first-run behavior of the test split."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    good_dir = Path(good_dir)
    bad_dir = Path(bad_dir)
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(project_dir)

    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "good_dir": str(good_dir),
            "bad_dir": str(bad_dir),
            "category": bad_dir.name,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "seed": seed,
            "images": {},
        }

    images = manifest["images"]
    is_first_run = len(images) == 0

    new_good = [n for n in _list_images(good_dir) if f"good/{n}" not in images]
    new_bad = [n for n in _list_images(bad_dir) if f"bad/{n}" not in images]

    good_assignments = _stratified_assign(new_good, ratios, seed, is_first_run)
    bad_assignments = _stratified_assign(new_bad, ratios, seed + 1, is_first_run)

    for name, split in good_assignments.items():
        images[f"good/{name}"] = {"class": "good", "split": split}
    for name, split in bad_assignments.items():
        images[f"bad/{name}"] = {"class": "bad", "split": split}

    _write_json(path, manifest)
    return manifest


def get_split_filepaths(manifest, split):
    """Returns {"good": [Path, ...], "bad": [Path, ...]} for images assigned to `split`."""
    good_dir = Path(manifest["good_dir"])
    bad_dir = Path(manifest["bad_dir"])
    result = {"good": [], "bad": []}
    for key, info in manifest["images"].items():
        if info["split"] != split:
            continue
        filename = key.split("/", 1)[1]
        base_dir = good_dir if info["class"] == "good" else bad_dir
        result[info["class"]].append(base_dir / filename)
    return result


def relabel_image(manifest, project_dir, key, new_class):
    """Move an image between good_dir/bad_dir, swap its manifest key/class, and
    persist the manifest immediately - this is what "an active-learning round moved
    an image between classes" (see module docstring) concretely means. `split` is
    preserved unchanged, never re-rolled, so a relabeled image can't accidentally end
    up in - or drop out of - the frozen test set.

    Mutates `manifest` in place and returns the new key (e.g. "bad/filename.png").

    Raises:
        KeyError: `key` is not in manifest["images"].
        ValueError: `new_class` isn't "good"/"bad", equals the image's current class
            (callers should skip that case rather than call this as a no-op), or the
            image's split is "test" (the frozen test set must never be touched here).
        FileExistsError: a different image already occupies the target filename in
            the destination directory - refuses to silently overwrite it.
    """
    if new_class not in ("good", "bad"):
        raise ValueError(f'new_class must be "good" or "bad", got {new_class!r}')

    info = manifest["images"][key]
    if info["split"] == "test":
        raise ValueError(f"refusing to relabel {key!r}: it's in the frozen test split")
    if info["class"] == new_class:
        raise ValueError(f"{key!r} is already class {new_class!r}")

    filename = key.split("/", 1)[1]
    good_dir = Path(manifest["good_dir"])
    bad_dir = Path(manifest["bad_dir"])
    old_dir = good_dir if info["class"] == "good" else bad_dir
    new_dir = bad_dir if new_class == "bad" else good_dir

    old_path = old_dir / filename
    new_path = new_dir / filename
    if new_path.exists():
        raise FileExistsError(f"can't relabel {key!r}: {new_path} already exists")

    shutil.move(str(old_path), str(new_path))

    del manifest["images"][key]
    new_key = f"{new_class}/{filename}"
    manifest["images"][new_key] = {"class": new_class, "split": info["split"]}

    save_manifest(manifest, project_dir)
    return new_key
