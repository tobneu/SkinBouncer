import pytest

from labeling_tool.settings import load_theme, save_theme


def test_load_theme_missing_file_returns_none(tmp_path):
    assert load_theme(tmp_path / "settings.json") is None


def test_save_then_load_theme_roundtrips(tmp_path):
    path = tmp_path / "nested" / "settings.json"
    save_theme("dark", path)
    assert load_theme(path) == "dark"


def test_save_theme_overwrites_previous_value(tmp_path):
    path = tmp_path / "settings.json"
    save_theme("dark", path)
    save_theme("light", path)
    assert load_theme(path) == "light"


def test_save_theme_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError):
        save_theme("blue", tmp_path / "settings.json")


def test_load_theme_ignores_corrupt_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("not json")
    assert load_theme(path) is None


def test_load_theme_ignores_unknown_stored_value(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"theme": "blue"}')
    assert load_theme(path) is None
