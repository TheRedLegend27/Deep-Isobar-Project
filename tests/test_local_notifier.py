"""Tests for the macOS local notifier (notifications/local_notifier.py)."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from deep_isobar.notifications import local_notifier


class TestEnabledGuards:
    def test_disabled_under_pytest(self):
        # PYTEST_CURRENT_TEST is set by pytest itself — the guard must see it.
        assert local_notifier._enabled() is False

    def test_disabled_on_non_darwin(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setattr(local_notifier.sys, "platform", "linux")
        assert local_notifier._enabled() is False

    def test_disabled_by_env_override(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setattr(local_notifier.sys, "platform", "darwin")
        monkeypatch.setenv("DEEP_ISOBAR_NO_LOCAL_NOTIFY", "1")
        assert local_notifier._enabled() is False

    def test_enabled_on_darwin_outside_pytest(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("DEEP_ISOBAR_NO_LOCAL_NOTIFY", raising=False)
        monkeypatch.setattr(local_notifier.sys, "platform", "darwin")
        assert local_notifier._enabled() is True


class TestAppleScriptQuote:
    def test_plain_text_unchanged(self):
        assert local_notifier._applescript_quote("hello") == "hello"

    def test_double_quotes_escaped(self):
        assert local_notifier._applescript_quote('a "b" c') == 'a \\"b\\" c'

    def test_backslashes_escaped_before_quotes(self):
        # A payload of \" must become \\\" — backslash pass must run first.
        assert local_notifier._applescript_quote('\\"') == '\\\\\\"'


class TestPostNotification:
    def test_noop_when_disabled(self):
        with patch.object(subprocess, "run") as run:
            local_notifier.post_notification("t", "m")
        run.assert_not_called()

    def test_invokes_osascript_when_enabled(self):
        with patch.object(local_notifier, "_enabled", return_value=True), \
             patch.object(subprocess, "run") as run:
            local_notifier.post_notification('Alert "now"', "body", sound=True)
        run.assert_called_once()
        argv = run.call_args.args[0]
        assert argv[0] == "osascript"
        script = argv[2]
        assert 'with title "Alert \\"now\\""' in script
        assert 'display notification "body"' in script
        assert 'sound name "Basso"' in script

    def test_no_sound_flag(self):
        with patch.object(local_notifier, "_enabled", return_value=True), \
             patch.object(subprocess, "run") as run:
            local_notifier.post_notification("t", "m", sound=False)
        assert "sound name" not in run.call_args.args[0][2]

    def test_never_raises_on_subprocess_failure(self):
        with patch.object(local_notifier, "_enabled", return_value=True), \
             patch.object(subprocess, "run", side_effect=OSError("no osascript")):
            local_notifier.post_notification("t", "m")  # must not raise
