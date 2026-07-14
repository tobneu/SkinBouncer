import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def find_threshold_for_recall(y_true, y_prob, recall_target=0.95, n_steps=200):
    """Grid-search the strictest (highest) threshold whose recall still meets recall_target.

    Intended to be run on a validation split, not train or test — see the project's
    active-learning design notes on avoiding threshold/test leakage.

    Returns a pandas Series with the chosen threshold, recall, precision and F1.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    thresholds = np.linspace(0.05, 0.95, n_steps)
    rows = []
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        rec = ((y_hat == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)
        pre = ((y_hat == 1) & (y_true == 1)).sum() / max((y_hat == 1).sum(), 1)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        rows.append({"threshold": t, "recall": rec, "precision": pre, "f1": f1})

    results_df = pd.DataFrame(rows)
    eligible = results_df[results_df["recall"] >= recall_target]
    if eligible.empty:
        raise ValueError(f"No threshold reaches recall >= {recall_target} on the given data.")
    return eligible.loc[eligible["threshold"].idxmax()]
