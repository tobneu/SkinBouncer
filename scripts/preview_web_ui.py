"""Render the labeling tool's web UI offscreen, without opening a window.

`pywebview[qt]` already brings PyQt6, and QWebEngineView lays out and paints into its
own backing store when WA_DontShowOnScreen is set - the same engine pywebview drives the
real app with. That makes the UI checkable two ways: as a PNG, and as JavaScript
evaluated against the loaded page (--eval), so pixel checks can be assertions rather
than eyeballing.

The page is loaded from a temporary copy of labeling_tool/web/ with a stub
window.pywebview.api injected, so any review state can be described without a session,
a trained model, or TensorFlow.

Usage:
    python scripts/preview_web_ui.py --out /tmp/ui.png
    python scripts/preview_web_ui.py --screen blind --theme light --out /tmp/blind.png
    python scripts/preview_web_ui.py --eval "document.title"

Never launch the real app just to look at it - that opens a window on the user's
desktop and can interrupt whatever they were doing in it.
"""

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "labeling_tool" / "web"

SCREENS = ("labeling", "blind", "active-learning", "done")


def _state_js(screen):
    """The get_state() payload each js_api class produces, as the frontend sees it.
    Mirrors labeling_tool/api.py - LabelingAPI sets no model-derived fields at all,
    BlindTestReviewAPI adds recorded_class and can_skip=false, and ActiveLearningAPI
    adds the ranking metadata plus the retrain/export panels."""
    base = {
        "done": False, "index": 12, "total": 40, "remaining": 28,
        "filename": "example.png",
    }
    if screen == "done":
        return {**base, "done": True, "index": 40, "remaining": 0, "filename": None,
                "image_data_uri": None, "can_retrain": True, "can_export": True,
                "run_comparison": None, "confusion_matrix": None, "test_curation": None}
    if screen == "labeling":
        return base
    if screen == "blind":
        return {**base, "recorded_class": "bad", "can_skip": False}
    return {
        **base,
        "recorded_class": "good", "predicted_prob": 0.732, "reason": "model disagrees",
        "can_retrain": True, "can_export": True,
        "run_comparison": {
            "current": {"val_auc": 0.883},
            "previous": [{"val_auc": 0.861, "pct_change": 2.6},
                         {"val_auc": 0.842, "pct_change": 4.9}],
        },
        "confusion_matrix": {"tp": 21, "tn": 22, "fp": 39, "fn": 1, "n": 62,
                             "recall": 21 / 22, "precision": 21 / 60, "accuracy": 43 / 62},
        "test_curation": {"reviewed": 0, "total": 62, "complete": False},
    }


def _default_skin():
    for folder in (ROOT / "sample_data" / "good", ROOT / "sample_data" / "bad_demo"):
        pngs = sorted(folder.glob("*.png")) if folder.is_dir() else []
        if pngs:
            return pngs[0]
    return None


def _build_page(screen, skin_path):
    """Copies the real web assets somewhere writable and drops a mock bridge in front of
    them, so index.html itself is rendered rather than a stand-in for it."""
    work = Path(tempfile.mkdtemp(prefix="skinbouncer-ui-"))
    shutil.copytree(WEB_DIR, work / "web")

    state = _state_js(screen)
    if state.get("filename") is not None and skin_path is not None:
        encoded = base64.b64encode(Path(skin_path).read_bytes()).decode("ascii")
        state["image_data_uri"] = f"data:image/png;base64,{encoded}"
        state["filename"] = Path(skin_path).name

    stub = f"""<script>
const STATE = {json.dumps(state)};
window.__errors = [];
window.addEventListener("error", (e) => window.__errors.push(String(e.message)));
window.addEventListener("unhandledrejection", (e) => window.__errors.push("rejection: " + e.reason));
window.pywebview = {{api: {{
  get_state: () => Promise.resolve(STATE),
  decide: () => Promise.resolve(STATE),
  retrain: () => Promise.resolve({{status: "started"}}),
  get_training_progress: () => Promise.resolve({{status: "idle"}}),
  export_detector: () => Promise.resolve({{category: "demo", threshold: 0.5,
                                           dest_dir: "06_Deployment/api/models/detectors/demo"}}),
}}}};
window.addEventListener("load", () => window.dispatchEvent(new Event("pywebviewready")));
</script>
"""
    index = work / "web" / "index.html"
    marker = '<script src="skin3d.js"></script>'
    html = index.read_text()
    if marker not in html:
        raise SystemExit(f"could not find {marker!r} in index.html - update this script")
    index.write_text(html.replace(marker, stub + "  " + marker))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen", choices=SCREENS, default="active-learning",
                        help="Which js_api's state shape to render (default: active-learning)")
    parser.add_argument("--theme", choices=("dark", "light", "system"), default="system")
    parser.add_argument("--skin", default=None, help="Skin PNG to show (default: one from sample_data)")
    parser.add_argument("--out", default=None, help="Write a screenshot here")
    parser.add_argument("--eval", dest="expression", default=None,
                        help="JavaScript to evaluate once loaded; result printed as JSON")
    parser.add_argument("--width", type=int, default=620)
    parser.add_argument("--height", type=int, default=1150)
    parser.add_argument("--settle-ms", type=int, default=1500,
                        help="Wait before capturing, so skins finish decoding")
    args = parser.parse_args()

    if args.out is None and args.expression is None:
        parser.error("nothing to do - pass --out, --eval, or both")

    # Chromium reads this when the web engine starts up, so it has to be set before Qt
    # is imported, not just before the QApplication exists.
    if args.theme != "system":
        flag = "1" if args.theme == "light" else "0"
        existing = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
            f"{existing} --blink-settings=preferredColorScheme={flag}".strip()
        )

    try:
        from PyQt6.QtCore import Qt, QTimer, QUrl
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWidgets import QApplication
    except ImportError as e:
        raise SystemExit(
            f"PyQt6 with QtWebEngine is required ({e}). It ships with the labeling-tool "
            f'extra: uv sync --all-extras, or pip install -e ".[labeling-tool]".'
        )

    page_path = _build_page(args.screen, args.skin or _default_skin())

    # QtWebEngine builds its own Chromium command line from argv and aborts outright if
    # it can't find a program name, so an empty list isn't an option here.
    app = QApplication(sys.argv[:1])
    view = QWebEngineView()
    view.resize(args.width, args.height)
    # Laid out and painted, but never mapped onto the desktop.
    view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    view.show()

    console = []
    view.page().javaScriptConsoleMessage = (
        lambda level, message, line, source: console.append(f"{message} ({source}:{line})")
    )
    exit_code = {"value": 0}

    def finish():
        if args.out:
            pixmap = QPixmap(view.size())
            view.render(pixmap)
            pixmap.save(str(args.out))
            print(f"wrote {args.out} ({pixmap.width()}x{pixmap.height()}, {args.screen}, {args.theme})")
        for line in console:
            print(f"console: {line}", file=sys.stderr)
        if console:
            exit_code["value"] = 1
        app.quit()

    def evaluated(value):
        print(json.dumps(value, indent=2, default=str))
        finish()

    def on_settled():
        if args.expression:
            view.page().runJavaScript(args.expression, evaluated)
        else:
            finish()

    def on_load(ok):
        if not ok:
            print(f"failed to load {page_path}", file=sys.stderr)
            exit_code["value"] = 1
            app.quit()
            return
        QTimer.singleShot(args.settle_ms, on_settled)

    view.loadFinished.connect(on_load)
    view.load(QUrl.fromLocalFile(str(page_path)))
    app.exec()

    shutil.rmtree(page_path.parent.parent, ignore_errors=True)
    sys.exit(exit_code["value"])


if __name__ == "__main__":
    main()
