"""CLI wrapper around labeling_tool.run_skinbouncer.

Opens directly on a project overview listing every detector project under
--projects-root, with a "New Project" wizard for creating new ones - the friendlier,
no-args entrypoint for anyone not reaching for the other scripts/run_*.py scripts and
their --project-dir flags directly.

Usage:
    python scripts/run_skinbouncer.py
    python scripts/run_skinbouncer.py --projects-root /path/to/other/projects
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labeling_tool import run_skinbouncer  # noqa: E402


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects-root", default=str(ROOT / "detector_projects"),
        help="Folder containing detector projects (default: detector_projects/)",
    )
    args = parser.parse_args()
    run_skinbouncer(args.projects_root)


if __name__ == "__main__":
    cli()
