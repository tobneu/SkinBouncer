"""Backs the project-overview screen: listing detector projects under a shared
projects-root folder, and cold-starting training for a brand-new project or an
existing-but-untrained one - the two ways a project can reach the review queue
without a trained checkpoint yet.

Kept separate from ActiveLearningSession (which requires a checkpoint to already
exist, see its own docstring) rather than folded into it, since "no project open yet"
and "a project open for review" are genuinely different states with almost no
overlapping behavior beyond both eventually training a model.
"""

import re
import threading
from pathlib import Path

from skinbouncer_core import (
    load_manifest,
    project_display_name,
    setup_detector_project,
    train_detector,
)

MANIFEST_FILENAME = "split_manifest.json"


def _slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "project"


def _unique_project_dir(projects_root, name):
    slug = _slugify(name)
    candidate = projects_root / slug
    n = 2
    while candidate.exists():
        candidate = projects_root / f"{slug}_{n}"
        n += 1
    return candidate


class ProjectOverviewSession:
    def __init__(self, projects_root):
        self.projects_root = Path(projects_root)
        self.training_progress = {"status": "idle"}
        self.trained_project_dir = None
        self._thread = None

    def list_projects(self):
        """Every immediate subdirectory of projects_root with a split_manifest.json -
        the same on-disk convention ActiveLearningSession itself requires."""
        if not self.projects_root.is_dir():
            return []
        projects = []
        for entry in sorted(self.projects_root.iterdir()):
            if not entry.is_dir() or not (entry / MANIFEST_FILENAME).exists():
                continue
            manifest = load_manifest(entry)
            trained = (entry / "model.keras").exists() and (entry / "threshold.json").exists()
            projects.append({
                "project_dir": str(entry),
                "display_name": project_display_name(manifest, entry),
                "trained": trained,
            })
        return projects

    def create_project(self, name, good_dir, bad_dir, epochs=50, batch_size=32):
        """Creates a brand-new project's split manifest, then cold-starts training on
        a background thread. Raises synchronously if the manifest itself can't be
        created (e.g. an empty folder) - that's cheap and worth failing fast on,
        before ever touching a background thread."""
        self.projects_root.mkdir(parents=True, exist_ok=True)
        project_dir = _unique_project_dir(self.projects_root, name)
        setup_detector_project(good_dir, bad_dir, project_dir, name=name)
        self._start_cold_start_training(project_dir, epochs, batch_size)

    def train_existing(self, project_dir, epochs=50, batch_size=32):
        """Same cold-start path as create_project, for a project whose manifest
        already exists (e.g. set up via the CLI) but has never been trained."""
        self._start_cold_start_training(Path(project_dir), epochs, batch_size)

    def _start_cold_start_training(self, project_dir, epochs, batch_size):
        if self.training_progress.get("status") == "running":
            raise RuntimeError("a project is already being created/trained")

        self.training_progress = {"status": "running", "epoch": 0, "epochs_total": epochs, "history": {}}
        self.trained_project_dir = None

        def on_epoch_end(epoch, logs):
            self.training_progress["epoch"] = epoch + 1
            history = self.training_progress["history"]
            for k, v in logs.items():
                history.setdefault(k, []).append(float(v))

        def worker():
            try:
                train_detector(project_dir, epochs=epochs, batch_size=batch_size, on_epoch_end=on_epoch_end)
                self.trained_project_dir = project_dir
                self.training_progress["status"] = "done"
            except Exception as e:
                self.training_progress["status"] = "error"
                self.training_progress["error"] = str(e)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
