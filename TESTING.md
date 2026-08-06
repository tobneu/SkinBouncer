# Testing

## Running tests

```powershell
pip install -e ".[dev]"
pytest
```

All tests live under `tests/` and use `pytest`. There's no CI configured yet, so run
this locally before opening a PR.

## Philosophy

Different parts of this project need different testing strategies - there's no single
approach that fits a folder-walking labeling tool, a pywebview GUI, and a CNN training
pipeline equally well.

### Pure logic: test directly, no mocking needed

Modules with no external dependencies (`labeling_tool/review_session.py`,
`skinbouncer_core/detector_project.py`) are tested by calling them directly against
real temp-directory fixtures (`tmp_path`). See `tests/test_review_session.py`.

### GUI (pywebview): skip automation, test the logic behind the bridge

There's no headless pywebview/webview automation in this repo, and no CI to run it in
anyway. Instead, the `js_api` adapter classes are kept intentionally thin (a few lines
of JSON marshaling) and unit-tested directly in Python, isolated from the actual window
- see `tests/test_labeling_api.py` testing `LabelingAPI` without ever launching
`webview.start()`. The visual/interactive side of a GUI slice is a HITL (human-in-the-
loop) manual verification step instead, called out explicitly in each labeling-tool
issue's acceptance criteria.

### ML training: mock stochastic/expensive paths, don't rely on real training runs

`skinbouncer_core/train.py` trains a real CNN, which is both slow (unsuitable to run
per-test at any real scale) and **stochastic** - whether a specific edge case triggers
(e.g. `find_threshold_for_recall` failing to reach a target recall) depends on the
probabilities a real training run happens to produce, not just on the code path taken.
Relying on a live run to exercise that branch would make the test flaky: it might pass
or fail depending on random initialization, hardware, or TensorFlow version, even
though the actual fallback *logic* never changed.

So `tests/test_train.py` uses two different strategies for two different things:

- **Wiring smoke test** (`test_train_detector_writes_all_outputs`): a real, tiny,
  1-epoch training run against synthetic fixture images (small enough to run in
  seconds), asserting the checkpoint/threshold/metrics files get written with the
  right shape and the checkpoint reloads cleanly. This is about "does the plumbing
  work end-to-end", not "does the model converge well" - accuracy/loss values aren't
  asserted on.
- **Deterministic branch coverage** (`test_train_detector_threshold_fallback_on_unreachable_recall`):
  mocks `find_threshold_for_recall` directly (`unittest.mock.patch`) to force the
  `ValueError` path, rather than trying to construct a dataset that happens to make
  real training fail. This tests the fallback-handling code itself - "if the search
  raises, do we still save the checkpoint and write a sane default?" - independent of
  whatever a real model would actually predict.

Pure helper functions with no ML in them (`_compute_sample_weights`, `_load_split_arrays`,
`_write_json`) are tested directly and cheaply, same as any other pure logic.

## Adding a new test

- If the code under test has no external dependencies (files, subprocesses, network,
  GUI, ML): test it directly.
- If it wraps something slow/stochastic (a real training run, a real HTTP call): mock
  the slow/stochastic part specifically, and keep a small number of real "does this
  actually work end-to-end" smoke tests separate from tests that need to be
  deterministic every run.
- If it's inherently visual/interactive (a GUI window): keep the thing that needs
  automation as thin as possible, unit-test the logic behind it, and rely on manual
  verification for the part that can't reasonably be automated yet.
