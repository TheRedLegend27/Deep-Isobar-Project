"""Native macOS Notification Center alerts for Deep Isobar.

The Discord notifier is a silent no-op unless ``DISCORD_WEBHOOK_URL`` is
set — which it deliberately isn't on this deployment.  During the
2026-08-09..14 outage every red ops_health embed and the kill-switch
engagement notice were dropped on the floor, and the frozen system went
unnoticed for days.  This module is the fallback channel that cannot be
unconfigured: a native macOS notification via ``osascript``, aimed at the
machine the operator is actually sitting at.

Reserved for CRITICAL events only (red ops_health invariants, kill-switch
engagement) so it never becomes noise worth ignoring:

    from deep_isobar.notifications.local_notifier import post_notification
    post_notification("Kill switch ENGAGED", "reason: daily_loss")

Silently no-ops on non-macOS platforms, under pytest, or when
``DEEP_ISOBAR_NO_LOCAL_NOTIFY`` is set.  Never raises — same contract as
``post_embed``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

_OSASCRIPT_TIMEOUT_S = 10


def _enabled() -> bool:
    if sys.platform != "darwin":
        return False
    if os.environ.get("DEEP_ISOBAR_NO_LOCAL_NOTIFY"):
        return False
    # The test suite exercises engage()/post_alarm_embed() directly; a full
    # run must not spray real notifications at the operator.
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return True


def _applescript_quote(text: str) -> str:
    """Escape *text* for inclusion inside an AppleScript double-quoted string."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def post_notification(title: str, message: str = "", sound: bool = True) -> None:
    """Show a macOS Notification Center alert.

    Args:
        title: Bold heading of the notification.
        message: Body text (keep it to one or two lines — Notification
            Center truncates aggressively).
        sound: Play the system "Basso" error sound so a red alert is heard
            even when the screen isn't being watched.
    """
    if not _enabled():
        return

    script = (
        f'display notification "{_applescript_quote(message)}" '
        f'with title "{_applescript_quote(title)}"'
    )
    if sound:
        script += ' sound name "Basso"'

    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=_OSASCRIPT_TIMEOUT_S,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 — alerting must never crash the caller
        logger.warning("local notification failed: %s", exc)
