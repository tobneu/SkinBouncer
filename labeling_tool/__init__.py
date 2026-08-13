from labeling_tool.active_learning_session import ActiveLearningSession
from labeling_tool.api import ActiveLearningAPI, BlindTestReviewAPI, LabelingAPI
from labeling_tool.blind_test_review_session import BlindTestReviewSession
from labeling_tool.review_session import ReviewSession

# The three window entrypoints live behind a lazy attribute because importing them pulls
# in pywebview, which only the `labeling-tool` extra installs. Everything above is plain
# logic with no GUI dependency, and importing the package must not require a GUI toolkit
# to reach it - that's what lets the tests exercise the js_api adapters without a window,
# and the headless deployment image import this package at all.
_LAZY_ENTRYPOINTS = {
    "main": "labeling_tool.app",
    "run_active_learning_queue": "labeling_tool.active_learning_app",
    "run_blind_test_review": "labeling_tool.blind_test_review_app",
}


def __getattr__(name):
    module_name = _LAZY_ENTRYPOINTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), "main")


def __dir__():
    return sorted(__all__)


__all__ = [
    "ReviewSession",
    "LabelingAPI",
    "main",
    "ActiveLearningSession",
    "ActiveLearningAPI",
    "run_active_learning_queue",
    "BlindTestReviewSession",
    "BlindTestReviewAPI",
    "run_blind_test_review",
]
