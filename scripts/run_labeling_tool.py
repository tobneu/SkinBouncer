"""CLI wrapper around labeling_tool.app.main.

Usage:
    python scripts/run_labeling_tool.py --folder sample_data/bad_demo
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labeling_tool import main  # noqa: E402


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, help="Folder of images to review")
    parser.add_argument("--good-subdir", default="good", help="Subfolder good images are moved to")
    parser.add_argument("--bad-subdir", default="bad", help="Subfolder bad images are moved to")
    parser.add_argument("--skip-subdir", default="skip", help="Subfolder skipped images are moved to")
    args = parser.parse_args()

    main(
        args.folder,
        good_subdir=args.good_subdir,
        bad_subdir=args.bad_subdir,
        skip_subdir=args.skip_subdir,
    )


if __name__ == "__main__":
    cli()
