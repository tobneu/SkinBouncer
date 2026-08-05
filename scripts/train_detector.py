"""CLI wrapper around skinbouncer_core.train_detector.

Usage:
    python scripts/train_detector.py --project-dir detector_projects/bad_demo
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from skinbouncer_core import train_detector  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True, help="Detector project dir (must contain split_manifest.json)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--recall-target", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the split manifest's seed")
    args = parser.parse_args()

    result = train_detector(
        project_dir=args.project_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        recall_target=args.recall_target,
        seed=args.seed,
    )

    print(f"\nTraining finished after {result['epochs_run']} epochs.")
    print(f"Checkpoint: {result['model_path']}")

    vm = result["val_metrics"]
    print(
        f"Val metrics: loss={vm['loss']:.4f}  accuracy={vm['accuracy']:.4f}  "
        f"precision={vm['precision']:.4f}  recall={vm['recall']:.4f}  auc={vm['auc']:.4f}"
    )

    ti = result["threshold_info"]
    if result["threshold_error"]:
        print(f"Threshold search FAILED ({result['threshold_error']}) - using fallback threshold={ti['threshold']}")
    else:
        print(
            f"Chosen threshold: {ti['threshold']:.4f} "
            f"(recall={ti['recall']:.4f}, precision={ti['precision']:.4f}, f1={ti['f1']:.4f})"
        )

    print(f"Wrote {result['threshold_path']} and {result['metrics_path']}")


if __name__ == "__main__":
    main()
