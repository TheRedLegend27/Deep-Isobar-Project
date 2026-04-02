"""Discord webhook notifier for Deep Isobar.

Sends rich embeds to a Discord channel via a webhook URL.  Uses only the
standard library (``urllib``) — no additional pip dependencies required.

Configuration
-------------
Set ``DISCORD_WEBHOOK_URL`` in your ``.env`` file::

    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/<id>/<token>

If the variable is absent the notifier logs a one-time warning and all
``post_embed`` calls become silent no-ops — the main scripts never crash
because of a missing or mis-configured webhook.

Colours (pass as the ``color`` argument)
-----------------------------------------
``COLOR_GREEN``  0x1D9E75 — win / success
``COLOR_RED``    0xE24B4A — loss / error
``COLOR_AMBER``  0xBA7517 — open position / warning
``COLOR_BLUE``   0x378ADD — info / neutral
``COLOR_GRAY``   0x888780 — no-signal / muted
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public colour constants
# ---------------------------------------------------------------------------

COLOR_GREEN = 0x1D9E75
COLOR_RED   = 0xE24B4A
COLOR_AMBER = 0xBA7517
COLOR_BLUE  = 0x378ADD
COLOR_GRAY  = 0x888780

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_webhook_url: str | None = None
_webhook_checked: bool = False


def _get_webhook_url() -> str | None:
    """Return the webhook URL, checking the environment exactly once."""
    global _webhook_url, _webhook_checked
    if not _webhook_checked:
        _webhook_url = os.environ.get("DISCORD_WEBHOOK_URL") or None
        if not _webhook_url:
            logger.warning(
                "DISCORD_WEBHOOK_URL not set — Discord notifications disabled."
            )
        _webhook_checked = True
    return _webhook_url


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def post_embed(
    title: str,
    description: str = "",
    color: int = COLOR_BLUE,
    fields: Sequence[dict] = (),
) -> None:
    """POST a Discord embed to the configured webhook.

    Args:
        title: Bold heading shown at the top of the embed.
        description: Optional body text (supports Discord markdown).
        color: Left-border colour as a 24-bit integer.
        fields: Sequence of ``{"name": str, "value": str, "inline": bool}``
            dicts.  ``inline`` defaults to ``True`` when omitted.

    The function returns immediately and logs a warning on HTTP errors rather
    than raising, so callers never need to wrap it in try/except.
    """
    url = _get_webhook_url()
    if not url:
        return

    normalised_fields = [
        {
            "name":   str(f.get("name", "")),
            "value":  str(f.get("value", "")),
            "inline": bool(f.get("inline", True)),
        }
        for f in fields
    ]

    payload = json.dumps(
        {
            "embeds": [
                {
                    "title":       title,
                    "description": description,
                    "color":       color,
                    "fields":      normalised_fields,
                }
            ]
        }
    ).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "DiscordBot (deep-isobar, 1.0)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                logger.warning(
                    "Discord webhook returned unexpected status %d", resp.status
                )
    except urllib.error.HTTPError as exc:
        logger.warning("Discord webhook HTTP error %d: %s", exc.code, exc.reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Discord webhook failed: %s", exc)
