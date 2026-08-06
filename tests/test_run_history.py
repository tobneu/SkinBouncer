import json

import pytest

from skinbouncer_core.run_history import append_run_history, compare_runs


def _entry(val_auc, warm_start=True, timestamp="2026-01-01T00:00:00+00:00"):
    return {
        "timestamp": timestamp,
        "warm_start": warm_start,
        "epochs_run": 1,
        "train": {"loss": 0.1, "accuracy": 0.9, "precision": 0.9, "recall": 0.9, "auc": val_auc},
        "val": {"loss": 0.1, "accuracy": 0.9, "precision": 0.9, "recall": 0.9, "auc": val_auc},
        "threshold_search": {"threshold": 0.5, "recall": 0.9, "precision": 0.9, "f1": 0.9, "recall_target": 0.95, "error": None},
    }


def test_append_run_history_creates_file_on_first_call(tmp_path):
    append_run_history(tmp_path, _entry(0.9))

    data = json.loads((tmp_path / "training_runs.json").read_text())
    assert len(data["runs"]) == 1
    assert data["runs"][0]["val"]["auc"] == pytest.approx(0.9)


def test_append_run_history_appends_not_overwrites(tmp_path):
    append_run_history(tmp_path, _entry(0.9))
    append_run_history(tmp_path, _entry(0.92))

    data = json.loads((tmp_path / "training_runs.json").read_text())
    assert [r["val"]["auc"] for r in data["runs"]] == pytest.approx([0.9, 0.92])


def test_compare_runs_returns_none_when_no_history(tmp_path):
    assert compare_runs(tmp_path) is None


def test_compare_runs_empty_previous_on_first_run(tmp_path):
    append_run_history(tmp_path, _entry(0.9))

    comparison = compare_runs(tmp_path)

    assert comparison["current"]["val_auc"] == pytest.approx(0.9)
    assert comparison["previous"] == []


def test_compare_runs_most_recent_first_with_correct_pct_change(tmp_path):
    append_run_history(tmp_path, _entry(0.80))
    append_run_history(tmp_path, _entry(0.88))
    append_run_history(tmp_path, _entry(0.90))

    comparison = compare_runs(tmp_path)

    assert comparison["current"]["val_auc"] == pytest.approx(0.90)
    assert [r["val_auc"] for r in comparison["previous"]] == pytest.approx([0.88, 0.80])
    assert comparison["previous"][0]["pct_change"] == pytest.approx((0.90 - 0.88) / 0.88 * 100)
    assert comparison["previous"][1]["pct_change"] == pytest.approx((0.90 - 0.80) / 0.80 * 100)


def test_compare_runs_caps_at_n_previous_runs(tmp_path):
    for auc in (0.5, 0.6, 0.7, 0.8, 0.85, 0.87, 0.9):
        append_run_history(tmp_path, _entry(auc))

    comparison = compare_runs(tmp_path, n=3)

    assert len(comparison["previous"]) == 3
    assert [r["val_auc"] for r in comparison["previous"]] == pytest.approx([0.87, 0.85, 0.8])
