"""CLI wrapper around labeling_tool.run_blind_test_review.

Walks the frozen test partition exactly once for manual confirm/correct, with no
model prediction or confidence shown - keeps the test set an independent ground
truth. Progress persists across relaunches.

Usage:
    python scripts/run_blind_test_review.py --project-dir detector_projects/bad_demo
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labeling_tool import run_blind_test_review  # noqa: E402


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", required=True,
        help="Detector project dir (must contain split_manifest.json)",
    )
    args = parser.parse_args()

    try:
        run_blind_test_review(args.project_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
