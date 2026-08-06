import os
from pathlib import Path

import webview

from labeling_tool.active_learning_session import ActiveLearningSession
from labeling_tool.api import ActiveLearningAPI

WEB_DIR = Path(__file__).parent / "web"


def main(project_dir):
    session = ActiveLearningSession(Path(project_dir))
    api = ActiveLearningAPI(session)

    print(f"Loaded {session.total()} train/val images ranked by suspicion.")

    webview.create_window(
        "SkinBouncer Labeling Tool - Active Learning",
        url=str(WEB_DIR / "index.html"),
        js_api=api,
        width=1280,
        height=720,
    )
    webview.start(gui="qt", debug=os.environ.get("SKINBOUNCER_DEBUG") == "1")

    print(
        f"Reviewed {session.index} / {session.total()} images this session "
        f"({session.relabel_count} relabeled)."
    )
