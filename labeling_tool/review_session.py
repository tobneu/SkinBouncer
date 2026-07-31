"""Walks a flat folder of images one at a time, moving each into a good/bad/skip
subfolder as the user decides. Folder location IS the label - no sidecar label file -
matching the convention `skinbouncer_core/detector_project.py` uses for good_dir/bad_dir.

Skipped images move to their own subfolder (not left in place) so that relaunching on
the same folder can tell "not yet reviewed" apart from "reviewed and skipped" just from
what's left in the root - the session re-globs the folder fresh on every run.
"""

import shutil
from pathlib import Path

ACTIONS = ("good", "bad", "skip")


class ReviewSession:
    def __init__(self, folder, good_subdir="good", bad_subdir="bad", skip_subdir="skip"):
        self.folder = Path(folder)
        self.items = sorted(self.folder.glob("*.png"))
        self.index = 0
        self._target_dirs = {
            "good": self.folder / good_subdir,
            "bad": self.folder / bad_subdir,
            "skip": self.folder / skip_subdir,
        }

    def total(self):
        return len(self.items)

    def remaining(self):
        return len(self.items) - self.index

    def is_done(self):
        return self.index >= len(self.items)

    def current_path(self):
        if self.is_done():
            return None
        return self.items[self.index]

    def decide(self, action):
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
        path = self.current_path()
        if path is None:
            raise IndexError("no image left to decide on")

        target_dir = self._target_dirs[action]
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target_dir / path.name))
        self.index += 1
