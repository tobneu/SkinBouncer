"""CLI wrapper around labeling_tool.run_active_learning_queue.

Ranks a detector project's train+val images by how much the current checkpoint
disagrees with each image's recorded label, and walks them in that order so
relabeling effort goes where it matters most.

Usage:
    python scripts/run_active_learning_queue.py --project-dir detector_projects/bad_demo
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labeling_tool import run_active_learning_queue  # noqa: E402


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", required=True,
        help="Detector project dir (must contain split_manifest.json, model.keras and threshold.json)",
    )
    args = parser.parse_args()

    try:
        run_active_learning_queue(args.project_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
