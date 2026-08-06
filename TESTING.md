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

### Front-end: render it offscreen rather than trusting it by eye

The "rely on manual verification" advice above still holds for *judgement* calls - does
this read well, is this contrast comfortable - but it turns out rather more than that
can be checked automatically, without ever putting a window on someone's screen.

`pywebview[qt]` already brings PyQt6, and `QWebEngineView` will lay out and paint into
its own backing store with `WA_DontShowOnScreen` set. That gives two things: a PNG of
the real UI in the real engine, and `page().runJavaScript(...)` to read values back out.
Point it at a copy of `labeling_tool/web/index.html` with a stub `window.pywebview.api`
injected, and the whole app shell renders against whatever state you want to describe.

`scripts/preview_web_ui.py` does exactly that:

```bash
# a screenshot of any screen, in either theme
python scripts/preview_web_ui.py --screen active-learning --theme dark --out /tmp/ui.png

# or an assertion - exits non-zero if the page logged anything to the console
python scripts/preview_web_ui.py --screen blind --eval "window.__errors"
```

`--screen` picks which `js_api` state shape to mock (`labeling`, `blind`,
`active-learning`, `done`), so the same run covers what each screen is meant to show.

That's how `skin3d.js` was checked in: a generated skin texture with each of the six
faces painted a different hue turns "does the model look right" into an assertion -
sample a pixel, and the colour names which face the renderer chose. It caught three real
defects that reviewing the code did not: mirrored legacy limbs rendering inside-out, an
inverted pitch axis, and past-round bars in the run-comparison chart drawn in a colour
that is identical to their own background in dark mode.

Worth reaching for whenever a change is visual but its correctness is not a matter of
taste - geometry, colour contrast, which control is visible in which mode.

### Contracts between two halves of the pipeline: test the round trip, not each side

Some agreements aren't owned by either module that depends on them. `export_detector()`
writes a detector folder; the deployment API's `load_detectors()` reads one. Each side
has its own tests, and both can pass while the two disagree about a filename, a folder
layout or a JSON key - a mismatch that surfaces only as a deployed image with silently
zero detectors loaded, which looks identical to "nothing exported yet".

`test_api_loads_and_scores_a_detector_produced_by_export_detector` in
`tests/test_deployment_api.py` covers that seam by driving the whole round trip: train
a project somewhere unrelated, export it, then import the API against the exported
folder and assert it both lists the detector and scores with the exported threshold.

Worth writing when two components communicate through a **format** rather than a
function call - a directory layout, a file on disk, a serialized payload - because
that's exactly where no single unit test naturally sits.

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
