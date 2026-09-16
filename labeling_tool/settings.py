"""Small local settings store for the labeling tool GUI - currently just the manual
light/dark theme override described in the app shell. Global rather than per-project,
since the same person runs every screen from the one desktop app.
"""

import json
import os
import tempfile
from pathlib import Path

DEFAULT_SETTINGS_PATH = Path.home() / ".skinbouncer" / "settings.json"

VALID_THEMES = ("light", "dark")


def _read(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_theme(path=DEFAULT_SETTINGS_PATH):
    """Returns "light"/"dark" if a manual override was saved, or None to fall back to
    the OS-level prefers-color-scheme."""
    theme = _read(Path(path)).get("theme")
    return theme if theme in VALID_THEMES else None


def save_theme(theme, path=DEFAULT_SETTINGS_PATH):
    """Writes via a temp file + atomic rename, so a crash or interrupt mid-write can
    never leave a partially-written / corrupt settings file behind (same pattern as
    skinbouncer_core/detector_project.py's _write_json)."""
    if theme not in VALID_THEMES:
        raise ValueError(f"unknown theme: {theme!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read(path)
    data["theme"] = theme
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise
