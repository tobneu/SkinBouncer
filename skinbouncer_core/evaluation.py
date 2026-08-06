"""Scores a detector project's frozen test split - the one partition train_detector()
never touches - so its performance can be reported without influencing training,
warm-start retraining, or the active-learning queue's ranking in any way.
"""

from .train import _load_split_arrays


def evaluate_confusion_matrix(manifest, model, threshold):
    """Scores every image in the frozen test split and tallies a 2x2 confusion
    matrix. good=0/bad=1 matches _load_split_arrays' convention; `score > threshold`
    matches 06_Deployment/api/main.py's risk convention, so this reports the same
    accept/reject decision the deployed detector would make.

    Returns None if the project's test split is empty, else plain-int
    {"tp", "tn", "fp", "fn", "n"} (n = tp+tn+fp+fn):
        tp = actually bad, predicted bad   (correctly caught)
        tn = actually good, predicted good (correctly passed)
        fp = actually good, predicted bad  (false alarm)
        fn = actually bad, predicted good  (missed)
    """
    X, y = _load_split_arrays(manifest, "test")
    if len(y) == 0:
        return None

    probs = model.predict(X, verbose=0).ravel()
    predicted_bad = probs > threshold
    actual_bad = y == 1.0

    return {
        "tp": int((predicted_bad & actual_bad).sum()),
        "tn": int((~predicted_bad & ~actual_bad).sum()),
        "fp": int((predicted_bad & ~actual_bad).sum()),
        "fn": int((~predicted_bad & actual_bad).sum()),
        "n": int(len(y)),
    }
