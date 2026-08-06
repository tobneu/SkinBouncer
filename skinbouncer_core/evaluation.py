"""Scores a detector project's frozen test split - the one partition train_detector()
never touches - so its performance can be reported without influencing training,
warm-start retraining, or the active-learning queue's ranking in any way.

These numbers are what an export decision should be judged on: the threshold itself is
tuned on val (see threshold.find_threshold_for_recall), so val metrics are optimistic by
construction, while test has never been fit on *or* tuned against.
"""

from .train import _load_split_arrays


def _rate(numerator, denominator):
    """None rather than 0.0 for an undefined rate - a test split with no bad images at
    all has no recall to report, which is a different statement from "recall is zero".
    Callers render None as "n/a" instead of a misleading percentage."""
    if denominator == 0:
        return None
    return numerator / denominator


def evaluate_confusion_matrix(manifest, model, threshold):
    """Scores every image in the frozen test split and tallies a 2x2 confusion
    matrix. good=0/bad=1 matches _load_split_arrays' convention; `score > threshold`
    matches 06_Deployment/api/main.py's risk convention, so this reports the same
    accept/reject decision the deployed detector would make.

    Returns None if the project's test split is empty, else a dict of plain-int counts
    (n = tp+tn+fp+fn) plus the float-or-None rates derived from them:
        tp = actually bad, predicted bad   (correctly caught)
        tn = actually good, predicted good (correctly passed)
        fp = actually good, predicted bad  (false alarm)
        fn = actually bad, predicted good  (missed)

        recall    = tp / (tp + fn)  share of bad images the detector catches
        precision = tp / (tp + fp)  share of flagged images that really are bad
        accuracy  = (tp + tn) / n   share of all decisions that were correct

    Counts are cast to plain ints (not numpy scalars) because this dict crosses the
    pywebview js_api bridge as JSON.
    """
    X, y = _load_split_arrays(manifest, "test")
    if len(y) == 0:
        return None

    probs = model.predict(X, verbose=0).ravel()
    predicted_bad = probs > threshold
    actual_bad = y == 1.0

    tp = int((predicted_bad & actual_bad).sum())
    tn = int((~predicted_bad & ~actual_bad).sum())
    fp = int((predicted_bad & ~actual_bad).sum())
    fn = int((~predicted_bad & actual_bad).sum())
    n = int(len(y))

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": n,
        "recall": _rate(tp, tp + fn),
        "precision": _rate(tp, tp + fp),
        "accuracy": _rate(tp + tn, n),
    }


def curation_status(manifest):
    """How far the frozen test split has been through blind review (see
    labeling_tool.blind_test_review_session, which writes the "reviewed" flag).

    Returns {"reviewed": int, "total": int, "complete": bool}. Until complete, the
    labels the confusion matrix is scored against are still just whatever the original
    bulk sort produced, so its numbers are only as trustworthy as that sort was - which
    is exactly the caveat an export decision needs surfaced alongside them.
    """
    test_entries = [info for info in manifest["images"].values() if info["split"] == "test"]
    total = len(test_entries)
    reviewed = sum(1 for info in test_entries if info.get("reviewed"))
    return {"reviewed": reviewed, "total": total, "complete": total > 0 and reviewed == total}
