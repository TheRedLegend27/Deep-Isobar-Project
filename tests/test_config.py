from deep_isobar.config import load_settings


def test_load_settings():
    settings = load_settings()
    assert "project" in settings