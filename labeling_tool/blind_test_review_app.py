import os
from pathlib import Path

import webview

from labeling_tool.api import BlindTestReviewAPI
from labeling_tool.blind_test_review_session import BlindTestReviewSession

WEB_DIR = Path(__file__).parent / "web"


def main(project_dir):
    session = BlindTestReviewSession(Path(project_dir))
    api = BlindTestReviewAPI(session)

    print(f"Test set: {session.index} / {session.total()} already reviewed.")

    webview.create_window(
        "SkinBouncer Labeling Tool - Blind Test Review",
        url=str(WEB_DIR / "index.html"),
        js_api=api,
        width=1280,
        height=720,
    )
    webview.start(gui="qt", debug=os.environ.get("SKINBOUNCER_DEBUG") == "1")

    print(f"Test set: {session.index} / {session.total()} reviewed ({session.remaining()} remaining).")
