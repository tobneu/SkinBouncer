"""CLI wrapper around skinbouncer_core.setup_detector_project.

Usage:
    python scripts/setup_detector_project.py --good sample_data/good \
        --bad sample_data/bad_demo --project-dir detector_projects/bad_demo
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from skinbouncer_core import setup_detector_project  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good", required=True, help="Path to the good/ image folder")
    parser.add_argument("--bad", required=True, help="Path to the bad/<category>/ image folder")
    parser.add_argument("--project-dir", required=True, help="Where to write split_manifest.json")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=67)
    args = parser.parse_args()

    manifest = setup_detector_project(
        good_dir=args.good,
        bad_dir=args.bad,
        project_dir=args.project_dir,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )

    counts = {}
    for info in manifest["images"].values():
        key = (info["class"], info["split"])
        counts[key] = counts.get(key, 0) + 1

    print(f"Manifest written to {Path(args.project_dir) / 'split_manifest.json'}")
    for cls in ("good", "bad"):
        for split in ("train", "val", "test"):
            print(f"  {cls}/{split}: {counts.get((cls, split), 0)}")


if __name__ == "__main__":
    main()
