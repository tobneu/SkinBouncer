"""Ranks a detector project's train+val images by how much the current checkpoint's
prediction diverges from each image's recorded class, and walks them in that order -
the active-learning counterpart to review_session.ReviewSession's plain folder walk.

The frozen test split is never loaded, scored, or shown here - see
skinbouncer_core.detector_project's module docstring on why test must stay untouched.

Ranking: a single continuous "suspicion" score subsumes both signals the issue asks
for (near-threshold uncertainty and confident disagreement) as different magnitudes of
the same underlying quantity - how far the model's prediction diverges from the
recorded label:

    suspicion = prob        if recorded_class == "good"   (higher prob = more suspect)
    suspicion = 1 - prob    if recorded_class == "bad"     (lower prob = more suspect)

Sorting all train+val images descending by this score generalizes
04_Modeling/Modeling.ipynb cell 47's false-negative/false-positive-by-confidence
approach to also cover near-threshold cases that reference cell doesn't handle. A
human-readable `reason` is derived from the same inputs for display, independent of
the sort key, so both the "combined score" and "separate reasons" readings of the
issue's acceptance criteria are covered.

Known limitation, not fixed here: train images are somewhat leakage-biased toward
looking "confidently correct" since the model was fit on them, so this queue will
structurally undersurface mislabeled train images relative to equally-mislabeled val
images. Inherent to "run the current checkpoint over train+val" as specified.

Relabeling ("good"/"bad" when it differs from the recorded class) delegates to
skinbouncer_core.detector_project.relabel_image, which moves the file and persists the
manifest immediately - there is no in-session batching or undo. This queue does not
persist "already reviewed" state across relaunches: quitting and relaunching
recomputes the full ranked pool fresh from whatever the manifest/checkpoint currently
look like. That matches the issue's own "recomputed against whatever the current
checkpoint is" framing as a per-launch guarantee, not a per-decision one.
"""

import json
import threading
from pathlib import Path

from skinbouncer_core import (
    compare_runs,
    get_split_filepaths,
    load_images,
    load_manifest,
    load_model,
    relabel_image,
    train_detector,
)

ACTIONS = ("good", "bad", "skip")
BORDERLINE_BAND = 0.10


def _suspicion(recorded_class, prob):
    return prob if recorded_class == "good" else 1.0 - prob


def _reason(recorded_class, prob, threshold):
    predicted_class = "bad" if prob >= threshold else "good"
    if predicted_class != recorded_class:
        return "model disagrees"
    if abs(prob - threshold) <= BORDERLINE_BAND:
        return "borderline"
    return "confident agreement"


class ActiveLearningSession:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)
        project_dir = self.project_dir
        if not (project_dir / "split_manifest.json").exists():
            raise FileNotFoundError(
                f"{project_dir} is not a detector project directory (no split_manifest.json "
                f"there) - run scripts/setup_detector_project.py --project-dir {project_dir} first."
            )
        self.manifest = load_manifest(project_dir)

        model_path = project_dir / "model.keras"
        threshold_path = project_dir / "threshold.json"
        if not model_path.exists() or not threshold_path.exists():
            raise FileNotFoundError(
                f"{project_dir} has no trained checkpoint yet - run "
                f"scripts/train_detector.py --project-dir {project_dir} first."
            )
        self.model = load_model(model_path)
        self.threshold = json.loads(threshold_path.read_text())["threshold"]

        self.items = self._build_ranked_items()
        self.index = 0
        self.relabel_count = 0
        self.training_progress = {"status": "idle"}
        self.run_comparison = None
        self._retrain_thread = None

    def _build_ranked_items(self):
        entries = []  # (key, path, recorded_class)
        for split in ("train", "val"):
            filepaths = get_split_filepaths(self.manifest, split)
            for recorded_class in ("good", "bad"):
                for path in filepaths[recorded_class]:
                    entries.append((f"{recorded_class}/{path.name}", path, recorded_class))

        if not entries:
            return []

        X = load_images([path for _, path, _ in entries])
        probs = self.model.predict(X, verbose=0).ravel()

        items = []
        for (key, path, recorded_class), prob in zip(entries, probs):
            prob = float(prob)
            items.append({
                "key": key,
                "path": path,
                "recorded_class": recorded_class,
                "prob": prob,
                "suspicion": _suspicion(recorded_class, prob),
                "reason": _reason(recorded_class, prob, self.threshold),
            })

        items.sort(key=lambda item: item["suspicion"], reverse=True)
        return items

    def total(self):
        return len(self.items)

    def remaining(self):
        return len(self.items) - self.index

    def is_done(self):
        return self.index >= len(self.items)

    def current_item(self):
        if self.is_done():
            return None
        return self.items[self.index]

    def current_path(self):
        item = self.current_item()
        return item["path"] if item else None

    def decide(self, action):
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
        item = self.current_item()
        if item is None:
            raise IndexError("no image left to decide on")

        if action != "skip" and action != item["recorded_class"]:
            new_key = relabel_image(self.manifest, self.project_dir, item["key"], action)
            item["key"] = new_key
            item["recorded_class"] = action
            self.relabel_count += 1

        self.index += 1

    def retrain(self, epochs=50, batch_size=32, lr=1e-4, patience=5):
        """Fine-tune the current checkpoint against whatever the manifest looks like
        right now (including any relabels made so far this session), then re-rank the
        queue against the new checkpoint. lr/patience default lower than
        train_detector's own cold-start defaults - a warm-started model converges (or
        plateaus) faster than one starting from random init.

        Runs on a background thread and returns immediately - training_progress is
        updated live (polled by the UI via ActiveLearningAPI.get_training_progress())
        rather than this call blocking until training finishes. This sidesteps relying
        on pywebview servicing concurrent js_api calls: every call, including this one,
        stays fast, so there's nothing that needs to overlap."""
        if self.training_progress.get("status") == "running":
            raise RuntimeError("a retrain is already in progress")

        self.training_progress = {"status": "running", "epoch": 0, "epochs_total": epochs, "history": {}}

        def on_epoch_end(epoch, logs):
            self.training_progress["epoch"] = epoch + 1
            history = self.training_progress["history"]
            for k, v in logs.items():
                history.setdefault(k, []).append(float(v))

        def worker():
            try:
                train_detector(self.project_dir, epochs=epochs, batch_size=batch_size,
                                lr=lr, patience=patience, warm_start=True, on_epoch_end=on_epoch_end)
                self.model = load_model(self.project_dir / "model.keras")
                self.threshold = json.loads((self.project_dir / "threshold.json").read_text())["threshold"]
                self.items = self._build_ranked_items()
                self.index = 0
                self.run_comparison = compare_runs(self.project_dir, n=5)
                self.training_progress["status"] = "done"
            except Exception as e:
                self.training_progress["status"] = "error"
                self.training_progress["error"] = str(e)

        self._retrain_thread = threading.Thread(target=worker, daemon=True)
        self._retrain_thread.start()
