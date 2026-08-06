"""Publishes a trained detector project into the folder the deployment image is built
from, so a finished detector reaches the API without any manual file copying.

The exported layout is exactly what 06_Deployment/api/main.py's load_detectors() scans
for - one subfolder per category containing model.keras + threshold.json - and is the
same folder 06_Deployment/build.sh bakes into the image. A detector is "deployed"
simply by having been exported before the next build; there is no separate registry.
"""

import json
import shutil
from pathlib import Path

from .detector_project import load_manifest

# Resolved from this file rather than the cwd, so exporting works the same from a
# script, a test, or the labeling tool's GUI regardless of where it was launched.
DEFAULT_DETECTORS_DIR = Path(__file__).resolve().parent.parent / "06_Deployment" / "api" / "models" / "detectors"


def export_detector(project_dir, detectors_dir=None):
    """Copy a project's trained checkpoint and its tuned threshold into
    <detectors_dir>/<category>/. Overwrites a previous export of the same category -
    re-exporting after another retrain round is the expected workflow, not an error.

    The category comes from the project's manifest (bad_dir's folder name), which is
    what the API reports each score under, so the exported folder name and the
    API's response key can't drift apart.

    Returns {"category", "dest_dir", "model_path", "threshold_path", "threshold"}.

    Raises FileNotFoundError if the project has no split_manifest.json, or no
    model.keras/threshold.json yet (i.e. it was never trained).
    """
    project_dir = Path(project_dir)
    manifest = load_manifest(project_dir)

    model_path = project_dir / "model.keras"
    threshold_path = project_dir / "threshold.json"
    for path in (model_path, threshold_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{project_dir} has no trained checkpoint to export ({path.name} missing) - "
                f"run scripts/train_detector.py --project-dir {project_dir} first."
            )

    category = manifest["category"]
    dest_dir = Path(detectors_dir if detectors_dir is not None else DEFAULT_DETECTORS_DIR) / category
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_model = dest_dir / "model.keras"
    dest_threshold = dest_dir / "threshold.json"
    shutil.copyfile(model_path, dest_model)
    shutil.copyfile(threshold_path, dest_threshold)

    return {
        "category": category,
        "dest_dir": str(dest_dir),
        "model_path": str(dest_model),
        "threshold_path": str(dest_threshold),
        "threshold": json.loads(threshold_path.read_text())["threshold"],
    }
