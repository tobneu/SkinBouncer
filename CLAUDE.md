# Working in this repo

## Never launch the GUI to look at it

`scripts/run_*.py` open a real pywebview window on the maintainer's desktop. One of
these was once launched from a terminal to "sanity-check imports", the maintainer saw a
familiar window and clicked Retrain, and killing the process interrupted training
mid-epoch and left a project's checkpoint out of sync with its threshold.

To see the UI, render it offscreen instead:

```bash
python scripts/preview_web_ui.py --screen active-learning --theme dark --out /tmp/ui.png
python scripts/preview_web_ui.py --screen blind --eval "window.__errors"
```

`--eval` runs JavaScript against the loaded page, so colour, geometry and per-screen
control visibility are assertions rather than opinions. See TESTING.md for when this is
worth reaching for. Taste - spacing, feel, default camera angle - still needs a human.

## Publish visual work as an artifact

A diff can't show whether a render is correct, and the maintainer is often in a remote
session where the terminal is the only channel. When work is visual, gather real
evidence, then publish it as an artifact alongside the PR - findings first, screenshots
as support. Never publish a polished page over checks that were never run.

## `detector_projects/` is real data, not fixtures

Those are the maintainer's own working projects with real images and trained
checkpoints. Tests build throwaway projects under `tmp_path`; nothing should read from
or write to `detector_projects/` without being asked. Exporting is the one exception,
and it only ever writes to the deployment detectors folder.

## Comments explain the code, not its history

Write comments as the code's own rationale - why this approach, what breaks otherwise.
Never reference a conversation, a request, or "what was asked for". A comment saying a
face's winding must not be swapped because the backface test reads it is useful; a
comment saying it was changed after review feedback is noise a year from now.

## Conventions worth matching

- `skinbouncer_core/` is importable library code; `labeling_tool/` is the GUI;
  `scripts/` holds thin CLI wrappers that add `ROOT` to `sys.path`.
- Public helpers must not start with `test_` - pytest collects them as tests wherever
  they're imported.
- Values crossing the pywebview bridge are JSON: cast numpy scalars to plain
  `int`/`float`, and avoid JS reserved words as `js_api` method names.
- Run `pytest` before opening a PR. There's no CI.
