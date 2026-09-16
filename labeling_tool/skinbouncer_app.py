import os
from pathlib import Path

import webview

from labeling_tool.api import SkinBouncerAPI

WEB_DIR = Path(__file__).parent / "web"


def main(projects_root):
    api = SkinBouncerAPI(projects_root)

    window = webview.create_window(
        "SkinBouncer",
        url=str(WEB_DIR / "index.html"),
        js_api=api,
        width=1280,
        height=800,
    )
    # js_api is bound at create_window() time, but nothing about pick_folder() needs
    # the Window before the first click - attaching it right after is soon enough.
    api.window = window

    webview.start(gui="qt", debug=os.environ.get("SKINBOUNCER_DEBUG") == "1")
