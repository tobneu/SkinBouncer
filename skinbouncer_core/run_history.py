"""Per-project training run history: a lightweight, append-only log of every
train_detector call's final metrics (not the model checkpoints themselves - those
are still overwritten in place, see train.py). Lets the labeling tool show how a
retrain round compares against recent previous rounds.
"""

import json
from pathlib import Path

from .detector_project import _write_json

HISTORY_FILENAME = "training_runs.json"


def _history_path(project_dir):
    return Path(project_dir) / HISTORY_FILENAME


def _load_runs(project_dir):
    path = _history_path(project_dir)
    if not path.exists():
        return []
    return json.loads(path.read_text())["runs"]


def append_run_history(project_dir, entry):
    """entry: {"timestamp", "warm_start", "epochs_run", "train", "val", "threshold_search"}
    (the same shape train_detector already writes to metrics.json, plus timestamp/warm_start)."""
    runs = _load_runs(project_dir)
    runs.append(entry)
    _write_json(_history_path(project_dir), {"runs": runs})


def compare_runs(project_dir, n=5):
    """Returns {"current": {"timestamp", "val_auc"}, "previous": [...]}, where "previous"
    holds up to n prior rounds (most-recent-first), each with its own val_auc and the
    percentage change from that round to the current one. Returns None if no run has
    ever been recorded for this project. "previous" is empty (but "current" still
    populated) when this is the first-ever recorded run."""
    runs = _load_runs(project_dir)
    if not runs:
        return None

    current = runs[-1]
    previous_runs = list(reversed(runs[-(n + 1):-1]))

    def pct_change(prev_auc, current_auc):
        if not prev_auc:
            return None
        return (current_auc - prev_auc) / prev_auc * 100

    current_auc = current["val"]["auc"]
    return {
        "current": {"timestamp": current["timestamp"], "val_auc": current_auc},
        "previous": [
            {
                "timestamp": run["timestamp"],
                "val_auc": run["val"]["auc"],
                "pct_change": pct_change(run["val"]["auc"], current_auc),
            }
            for run in previous_runs
        ],
    }
