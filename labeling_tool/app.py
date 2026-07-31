import os
from pathlib import Path

import webview

from labeling_tool.api import LabelingAPI
from labeling_tool.review_session import ReviewSession

WEB_DIR = Path(__file__).parent / "web"


def main(folder, good_subdir="good", bad_subdir="bad", skip_subdir="skip"):
    session = ReviewSession(folder, good_subdir=good_subdir, bad_subdir=bad_subdir, skip_subdir=skip_subdir)
    api = LabelingAPI(session)

    webview.create_window(
        "SkinBouncer Labeling Tool",
        url=str(WEB_DIR / "index.html"),
        js_api=api,
        width=1280,
        height=720,
    )
    webview.start(gui="qt", debug=os.environ.get("SKINBOUNCER_DEBUG") == "1")
