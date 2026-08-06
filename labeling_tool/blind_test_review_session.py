"""Walks the frozen test partition for manual confirm/correct, showing no model
prediction or confidence anywhere - this is what keeps the test set an independent
ground truth for the export-gate metrics it will later be used to compute. Each image
is reviewed exactly once overall, but the walk itself is safe to pause and resume
freely - quitting mid-session (or after any single decision) and relaunching later
picks up right where it left off, with nothing to redo.

Unlike ActiveLearningSession (which deliberately recomputes its ranked pool fresh on
every launch, per #12's design), progress here must survive relaunches: a "reviewed"
flag is written directly onto each test-split manifest entry immediately after every
decide() call, so total()/index reflect overall test-set completion rather than just
this session's walk order. Relaunching shows real cumulative progress instead of
resetting to 0, and already-reviewed images never reappear.

No trained checkpoint is required - this mode never loads a model, so test-set
curation can start as soon as a split manifest exists, independent of training.

There is no "skip" action here (see #10's resolved planning notes): every image
forces an explicit good/bad decision, matching "reviewed exactly once."
"""

from pathlib import Path

from skinbouncer_core import load_manifest, relabel_image, save_manifest

ACTIONS = ("good", "bad")


class BlindTestReviewSession:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        if not (self.project_dir / "split_manifest.json").exists():
            raise FileNotFoundError(
                f"{self.project_dir} is not a detector project directory (no "
                f"split_manifest.json there) - run scripts/setup_detector_project.py "
                f"--project-dir {self.project_dir} first."
            )
        self.manifest = load_manifest(self.project_dir)

        test_entries = [(k, info) for k, info in self.manifest["images"].items() if info["split"] == "test"]
        self._total = len(test_entries)
        already_reviewed = sum(1 for _, info in test_entries if info.get("reviewed"))
        pending_keys = sorted(k for k, info in test_entries if not info.get("reviewed"))

        good_dir = Path(self.manifest["good_dir"])
        bad_dir = Path(self.manifest["bad_dir"])
        self.items = []
        for key in pending_keys:
            recorded_class = self.manifest["images"][key]["class"]
            base_dir = good_dir if recorded_class == "good" else bad_dir
            filename = key.split("/", 1)[1]
            self.items.append({"key": key, "path": base_dir / filename, "recorded_class": recorded_class})

        self._pos = 0  # cursor into self.items, this launch's remaining pool
        self.index = already_reviewed  # overall progress counter, survives relaunches

    def total(self):
        return self._total

    def remaining(self):
        return self._total - self.index

    def is_done(self):
        return self._pos >= len(self.items)

    def current_item(self):
        if self.is_done():
            return None
        return self.items[self._pos]

    def current_path(self):
        item = self.current_item()
        return item["path"] if item else None

    def decide(self, action):
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
        item = self.current_item()
        if item is None:
            raise IndexError("no image left to decide on")

        key = item["key"]
        if action == item["recorded_class"]:
            self.manifest["images"][key]["reviewed"] = True
            save_manifest(self.manifest, self.project_dir)
        else:
            new_key = relabel_image(self.manifest, self.project_dir, key, action, allow_test=True)
            self.manifest["images"][new_key]["reviewed"] = True
            save_manifest(self.manifest, self.project_dir)

        self._pos += 1
        self.index += 1
