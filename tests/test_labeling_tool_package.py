"""Guards that importing `labeling_tool` does not require pywebview.

TESTING.md's whole GUI strategy rests on the js_api adapters being reachable without a
window, and the deployment image installs the package without the `labeling-tool` extra.
Both break the moment `labeling_tool/__init__.py` eagerly imports one of the `*_app`
modules again - which is easy to do by adding a re-export, and which only shows up on a
machine that doesn't happen to have pywebview installed.
"""

import importlib
import sys

import pytest

GUI_ENTRYPOINTS = ["main", "run_active_learning_queue", "run_blind_test_review"]


@pytest.fixture
def without_pywebview(monkeypatch):
    """Make `import webview` fail, and force labeling_tool to be imported afresh so the
    package's own import machinery actually runs again."""
    for name in list(sys.modules):
        if name == "labeling_tool" or name.startswith("labeling_tool."):
            monkeypatch.delitem(sys.modules, name)
    # A None entry in sys.modules is what the import system treats as "this failed".
    monkeypatch.setitem(sys.modules, "webview", None)


def test_package_imports_without_pywebview(without_pywebview):
    module = importlib.import_module("labeling_tool")

    # The GUI-free half has to be reachable, since that is what the tests use.
    assert module.ReviewSession is not None
    assert module.LabelingAPI is not None
    assert module.ActiveLearningSession is not None
    assert module.BlindTestReviewAPI is not None


def test_session_modules_import_without_pywebview(without_pywebview):
    for name in ["api", "review_session", "active_learning_session",
                 "blind_test_review_session"]:
        importlib.import_module(f"labeling_tool.{name}")


@pytest.mark.parametrize("entrypoint", GUI_ENTRYPOINTS)
def test_gui_entrypoints_are_lazy_but_still_wired(without_pywebview, entrypoint):
    """Reaching a window entrypoint is what pulls pywebview in - so with it unavailable
    the attribute must fail on access, not on package import."""
    module = importlib.import_module("labeling_tool")

    with pytest.raises(ImportError):
        getattr(module, entrypoint)


def test_unknown_attribute_still_raises_attribute_error(without_pywebview):
    module = importlib.import_module("labeling_tool")

    with pytest.raises(AttributeError):
        module.no_such_thing


def test_gui_entrypoints_resolve_when_pywebview_is_available():
    """The other half of laziness: with the extra installed, the entrypoints the
    scripts/run_*.py wrappers import must actually resolve."""
    pytest.importorskip("webview", reason="requires the labeling-tool extra")
    module = importlib.import_module("labeling_tool")

    for entrypoint in GUI_ENTRYPOINTS:
        assert callable(getattr(module, entrypoint))
