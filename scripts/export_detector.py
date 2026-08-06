"""CLI wrapper around skinbouncer_core.export_detector.

Copies a trained project's checkpoint + tuned threshold into the folder
06_Deployment/build.sh bakes into the API image, and reports how the detector scores
on its frozen test split so the export decision has numbers behind it.

Usage:
    python scripts/export_detector.py --project-dir detector_projects/bad_demo
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from skinbouncer_core import (  # noqa: E402
    DEFAULT_DETECTORS_DIR,
    curation_status,
    evaluate_confusion_matrix,
    export_detector,
    load_manifest,
    load_model,
)


def _pct(rate):
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", required=True,
        help="Detector project dir (must contain split_manifest.json, model.keras and threshold.json)",
    )
    parser.add_argument(
        "--detectors-dir", default=None,
        help=f"Where to export to (default: {DEFAULT_DETECTORS_DIR})",
    )
    parser.add_argument(
        "--skip-metrics", action="store_true",
        help="Export without scoring the test split first (skips loading TensorFlow)",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    try:
        manifest = load_manifest(project_dir)
    except FileNotFoundError:
        print(f"Error: {project_dir} is not a detector project directory (no split_manifest.json there).")
        sys.exit(1)

    if not args.skip_metrics:
        threshold_path = project_dir / "threshold.json"
        model_path = project_dir / "model.keras"
        if not model_path.exists() or not threshold_path.exists():
            print(f"Error: {project_dir} has no trained checkpoint yet - run scripts/train_detector.py first.")
            sys.exit(1)

        threshold = json.loads(threshold_path.read_text())["threshold"]
        cm = evaluate_confusion_matrix(manifest, load_model(model_path), threshold)
        if cm is None:
            print("Test split is empty - no held-out metrics to report.")
        else:
            print(f"\nTest split ({cm['n']} images), scored at threshold {threshold:.4f}:")
            print(f"  catches bad skins        {_pct(cm['recall'])}  ({cm['tp']} of {cm['tp'] + cm['fn']})")
            print(f"  flags that were correct  {_pct(cm['precision'])}  ({cm['tp']} of {cm['tp'] + cm['fp']})")
            print(f"  overall correct          {_pct(cm['accuracy'])}  ({cm['tp'] + cm['tn']} of {cm['n']})")
            print(f"  missed {cm['fn']} bad, false-alarmed on {cm['fp']} good")

    # Reported, not enforced: an uncurated test split makes the numbers above unvetted,
    # which is the operator's call to weigh - it doesn't make the checkpoint itself
    # unexportable, and re-exporting after finishing curation costs nothing.
    curation = curation_status(manifest)
    if curation["total"] and not curation["complete"]:
        print(
            f"\nWARNING: only {curation['reviewed']} of {curation['total']} test images have been "
            f"blind-reviewed. Any metrics above are scored against unconfirmed labels - run "
            f"scripts/run_blind_test_review.py --project-dir {project_dir} to confirm them."
        )

    try:
        result = export_detector(project_dir, detectors_dir=args.detectors_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nExported detector '{result['category']}' (threshold {result['threshold']:.4f}) to:")
    print(f"  {result['dest_dir']}")
    print("\nShip it with:\n  06_Deployment/build.sh")


if __name__ == "__main__":
    main()
