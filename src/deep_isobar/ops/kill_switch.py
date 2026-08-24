"""Kill switch for Deep Isobar — the single, unmissable stop for all live trading.

Motivated by the go-live safety build (2026-08): once real capital is on
the line, there must be one way to stop every order that a panicking human
at 2am can reach, that automated triggers can reach without a human, and
that keeps working even if the supervisor process is wedged.

File-based source of truth
---------------------------
The engaged/disengaged state lives at ``data/KILL_SWITCH``
(:data:`_KILL_SWITCH_PATH`) — not in memory, not in ``config/settings.yaml``.
**File presence alone means ENGAGED.**  A bare ``touch data/KILL_SWITCH``
from a terminal is a fully valid way to stop trading; no Python required.
The file's JSON content (written by :func:`engage`) carries the audit trail
— reason, source, timestamp — but is never required for the engaged
determination itself:

- No file → disengaged.
- File exists, readable, well-formed JSON → engaged, full detail available.
- File exists but empty/corrupt/unreadable → still engaged. Presence is
  authoritative; a human who touched an empty file gets exactly the
  protection they wanted.
- The existence check itself fails (permissions, disk error) → engaged.
  An error must never be interpreted as "no switch, trade away."

This makes the state trivially crash-safe: there is no in-memory cache to
lose on restart, and no code path returns "disengaged" except the one
where the file provably does not exist.

Independent layers
-------------------
Checked separately in three places so no single missed call re-opens
trading:

1. :func:`deep_isobar.trading.trade_execution.submit_live_trade` — the
   final guard immediately before the (Part 3) network call.
2. :func:`deep_isobar.trading.preflight.run_preflight` — an early per-city
   gate, checked even when the rest of preflight is disabled via config.
3. :func:`deep_isobar.research.paper_trade_session.main` — before any city
   is processed at all.

Automatic triggers
-------------------
``check_*_trigger`` functions below are pure: given the relevant numbers
and a configured limit, they return a reason string once the limit is
reached (and only then), or ``None``.  Callers own deciding *when* to
evaluate them and call :func:`engage` with the returned reason — see
:func:`evaluate_trading_triggers` for the default wiring, invoked from
``settle_paper_trades.py`` (daily loss / drawdown / consecutive losses) and
``ops/health.py`` (invariant breakage).  Engaging is instant and may be
fully automatic.  Releasing is never automatic — see :func:`release`.

CLI::

    python -m deep_isobar.ops.kill_switch --status
    python -m deep_isobar.ops.kill_switch --engage "reason"
    python -m deep_isobar.ops.kill_switch --release --confirm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from deep_isobar.config import get_project_root, get_setting
from deep_isobar.notifications.discord_notifier import COLOR_GREEN, COLOR_RED, post_embed
from deep_isobar.notifications.local_notifier import post_notification

logger = logging.getLogger(__name__)

_KILL_SWITCH_PATH: Path = get_project_root() / "data" / "KILL_SWITCH"
_HISTORY_LOG_PATH: Path = get_project_root() / "data" / "logs" / "kill_switch_history.jsonl"

_DEFAULT_MAX_CONSECUTIVE_LOSSES = 5


class KillSwitchEngagedError(RuntimeError):
    """Raised when an operation refuses to proceed because the switch is engaged."""


@dataclass(frozen=True)
class KillSwitchState:
    """Current kill-switch state as read from the sentinel file."""

    engaged: bool
    reason: str | None = None
    source: str | None = None
    engaged_at_utc: str | None = None
    # Human-readable note for cases with no structured reason/source, e.g.
    # "sentinel file present but empty (manual touch?)".
    detail: str = ""


# ---------------------------------------------------------------------------
# Core state — read fresh from disk every call, fail closed on any error.
# ---------------------------------------------------------------------------


def get_state() -> KillSwitchState:
    """Return the current kill-switch state.

    File absence is the *only* condition that yields ``engaged=False``.
    Every other outcome — a permissions error on the path check, a corrupt
    or empty file, unparseable JSON — yields ``engaged=True``.  Never let
    an error open the gate.
    """
    try:
        exists = _KILL_SWITCH_PATH.exists()
    except OSError as exc:
        logger.critical(
            "kill_switch: cannot determine state (%s) — failing closed (ENGAGED)", exc
        )
        return KillSwitchState(engaged=True, detail=f"path check failed: {exc}")

    if not exists:
        return KillSwitchState(engaged=False)

    try:
        raw = _KILL_SWITCH_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.critical(
            "kill_switch: sentinel file present but unreadable (%s) — still "
            "ENGAGED (presence alone is authoritative)", exc,
        )
        return KillSwitchState(engaged=True, detail=f"file unreadable: {exc}")

    if not raw:
        return KillSwitchState(
            engaged=True, detail="sentinel file present but empty (manual touch?)"
        )

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    except (ValueError, json.JSONDecodeError) as exc:
        logger.critical(
            "kill_switch: sentinel file present but corrupt (%s) — still "
            "ENGAGED (presence alone is authoritative)", exc,
        )
        return KillSwitchState(engaged=True, detail=f"corrupt state file: {exc}")

    return KillSwitchState(
        engaged=True,
        reason=data.get("reason"),
        source=data.get("source"),
        engaged_at_utc=data.get("engaged_at_utc"),
    )


def is_engaged() -> bool:
    """True if live order placement (and the wider trading pipeline) must
    be refused right now.

    Importable and callable without side effects — this is the single
    function every execution layer must call before doing anything
    irreversible.
    """
    return get_state().engaged


# ---------------------------------------------------------------------------
# Engage / release
# ---------------------------------------------------------------------------


def engage(reason: str, source: str) -> None:
    """Engage the kill switch immediately.

    Safe to call repeatedly (idempotent in effect) and safe to call from
    automated triggers with no human involved. Writes the sentinel file,
    logs at CRITICAL, appends an audit-log entry, and posts a red Discord
    embed (best-effort — never raises on notification failure).

    Args:
        reason: Human-readable explanation, ideally carrying the values
            that triggered it, e.g. ``"daily loss $134.20 exceeds limit
            $100.00"``.
        source: Who/what engaged it, e.g. ``"human_cli:kaden"``,
            ``"daily_loss_trigger"``, ``"ops_health_trigger"``.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {"reason": reason, "source": source, "engaged_at_utc": now_iso}

    _KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    _KILL_SWITCH_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.critical("KILL SWITCH ENGAGED — source=%s reason=%s", source, reason)
    _append_history({"event": "engage", **payload})
    try:
        post_embed(
            title="\U0001f6d1 KILL SWITCH ENGAGED",
            description="All live order placement is now refused.",
            color=COLOR_RED,
            fields=[
                {"name": "Source", "value": source},
                {"name": "Reason", "value": reason},
                {"name": "Engaged at (UTC)", "value": now_iso},
            ],
        )
    except Exception:  # noqa: BLE001
        logger.warning("kill_switch: could not post engage Discord embed", exc_info=True)
    # Local notification — the un-unconfigurable channel (Discord has no
    # webhook set on this deployment; see the 2026-08-09..14 incident).
    post_notification(
        title="\U0001f6d1 Deep Isobar: KILL SWITCH ENGAGED",
        message=f"{source}: {reason}",
    )


def release(released_by: str, confirm: bool) -> None:
    """Disengage the kill switch. Always a deliberate human action.

    Never called automatically anywhere in this codebase — only the CLI
    (with an explicit ``--confirm`` flag) calls this.

    Args:
        released_by: Identifies who is releasing it (human name/handle).
        confirm: Must be ``True``. This is not a default-on operation —
            the whole point of :func:`release` existing as a separate,
            explicit function is that nothing else can call it by accident.

    Raises:
        ValueError: If ``confirm`` is not ``True``, or ``released_by`` is
            blank. Neither is a recoverable state — fix the call.
    """
    if not confirm:
        raise ValueError(
            "release() refused: confirm=True is required — disengaging the "
            "kill switch is never automatic or accidental."
        )
    if not released_by or not released_by.strip():
        raise ValueError(
            "release() refused: released_by must identify who is releasing it"
        )

    prior = get_state()
    if not prior.engaged:
        logger.info("kill_switch: release() called but switch was not engaged — no-op")
        return

    prior_reason_display = prior.reason or prior.detail or "(unknown — see history log)"

    try:
        _KILL_SWITCH_PATH.unlink(missing_ok=True)
    except OSError as exc:
        logger.error("kill_switch: could not remove sentinel file: %s", exc)
        raise

    now_iso = datetime.now(timezone.utc).isoformat()
    logger.critical(
        "KILL SWITCH RELEASED by %s — acknowledging prior reason: %s "
        "(source=%s, engaged_at=%s)",
        released_by, prior_reason_display, prior.source, prior.engaged_at_utc,
    )
    _append_history({
        "event": "release",
        "released_by": released_by,
        "released_at_utc": now_iso,
        "prior_reason": prior.reason,
        "prior_source": prior.source,
        "prior_engaged_at_utc": prior.engaged_at_utc,
        "prior_detail": prior.detail,
    })
    try:
        post_embed(
            title="✅ Kill switch released",
            description=f"Released by {released_by}.",
            color=COLOR_GREEN,
            fields=[{"name": "Prior reason", "value": prior_reason_display}],
        )
    except Exception:  # noqa: BLE001
        logger.warning("kill_switch: could not post release Discord embed", exc_info=True)


def _append_history(entry: dict) -> None:
    """Best-effort append to the audit log; never raises."""
    try:
        _HISTORY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _HISTORY_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        logger.warning("kill_switch: could not write history log entry", exc_info=True)


# ---------------------------------------------------------------------------
# Automatic triggers — pure functions. Callers decide when to evaluate them
# and call engage() with the returned reason; nothing here has side effects.
# ---------------------------------------------------------------------------


def check_daily_loss_trigger(
    daily_realized_pnl_usd: float, max_daily_loss_usd: float,
) -> str | None:
    """Reason string once today's realized loss reaches the configured limit."""
    if max_daily_loss_usd <= 0:
        return None
    loss = -daily_realized_pnl_usd
    if loss >= max_daily_loss_usd:
        return (
            f"daily realized loss ${loss:.2f} reached the limit "
            f"${max_daily_loss_usd:.2f}"
        )
    return None


def check_drawdown_trigger(
    current_equity_usd: float,
    peak_equity_usd: float,
    bankroll_usd: float,
    max_drawdown_pct: float,
) -> str | None:
    """Reason string once drawdown-from-peak reaches *max_drawdown_pct* of bankroll."""
    if max_drawdown_pct <= 0 or bankroll_usd <= 0:
        return None
    drawdown_usd = peak_equity_usd - current_equity_usd
    if drawdown_usd <= 0:
        return None
    drawdown_pct = drawdown_usd / bankroll_usd
    if drawdown_pct >= max_drawdown_pct:
        return (
            f"drawdown ${drawdown_usd:.2f} ({drawdown_pct:.1%} of ${bankroll_usd:.2f} "
            f"bankroll) reached the limit {max_drawdown_pct:.1%}"
        )
    return None


def check_consecutive_losses_trigger(
    consecutive_losses: int,
    max_consecutive_losses: int = _DEFAULT_MAX_CONSECUTIVE_LOSSES,
) -> str | None:
    """Reason string once *consecutive_losses* reaches the configured limit."""
    if max_consecutive_losses <= 0:
        return None
    if consecutive_losses >= max_consecutive_losses:
        return (
            f"{consecutive_losses} consecutive losing trades reached the "
            f"limit {max_consecutive_losses}"
        )
    return None


def check_ops_health_trigger(broken_invariant_names: list[str]) -> str | None:
    """Reason string whenever any ops-health invariant is broken."""
    if broken_invariant_names:
        return (
            f"ops-health invariant(s) broken: {', '.join(broken_invariant_names)} — "
            "trading blind on a broken invariant is how small bugs become large losses"
        )
    return None


def check_reconciliation_trigger(
    local_value_usd: float, exchange_value_usd: float, tolerance_usd: float,
) -> str | None:
    """Reason string when local position/balance disagrees with the exchange.

    Ready for the Part 3 reconciliation job to call once live positions
    exist; not wired to any caller yet since there is no live exchange
    state to reconcile against until live execution ships.
    """
    mismatch = abs(local_value_usd - exchange_value_usd)
    if mismatch > tolerance_usd:
        return (
            f"local/exchange mismatch ${mismatch:.2f} exceeds tolerance "
            f"${tolerance_usd:.2f} (local=${local_value_usd:.2f} "
            f"exchange=${exchange_value_usd:.2f})"
        )
    return None


def evaluate_trading_triggers(
    *,
    daily_realized_pnl_usd: float | None = None,
    current_equity_usd: float | None = None,
    peak_equity_usd: float | None = None,
    consecutive_losses: int | None = None,
    broken_invariant_names: list[str] | None = None,
) -> list[str]:
    """Evaluate every trigger with inputs supplied, engaging on any hit.

    Each argument is independently optional — a caller that only has
    today's P&L (e.g. the settlement job) passes just
    ``daily_realized_pnl_usd``; a caller with the full picture passes
    everything. Thresholds come from ``risk.kill_switch`` in
    ``settings.yaml``; bankroll from ``risk.position_sizing.bankroll_usd``
    (the Part 2 smart sizer's single authoritative bankroll figure).

    Returns:
        The reasons that fired, in evaluation order (empty if none did).
        Each firing reason has already been passed to :func:`engage`.
    """
    bankroll_usd = float(get_setting("risk.position_sizing.bankroll_usd", 500.0))
    max_daily_loss = float(get_setting("risk.kill_switch.max_daily_loss_usd", 100.0))
    max_drawdown_pct = float(get_setting("risk.kill_switch.max_drawdown_pct", 0.20))
    max_consecutive = int(
        get_setting("risk.kill_switch.max_consecutive_losses", _DEFAULT_MAX_CONSECUTIVE_LOSSES)
    )

    fired: list[str] = []

    if daily_realized_pnl_usd is not None:
        reason = check_daily_loss_trigger(daily_realized_pnl_usd, max_daily_loss)
        if reason:
            engage(reason, source="daily_loss_trigger")
            fired.append(reason)

    if current_equity_usd is not None and peak_equity_usd is not None:
        reason = check_drawdown_trigger(
            current_equity_usd, peak_equity_usd, bankroll_usd, max_drawdown_pct
        )
        if reason:
            engage(reason, source="drawdown_trigger")
            fired.append(reason)

    if consecutive_losses is not None:
        reason = check_consecutive_losses_trigger(consecutive_losses, max_consecutive)
        if reason:
            engage(reason, source="consecutive_losses_trigger")
            fired.append(reason)

    if broken_invariant_names is not None:
        reason = check_ops_health_trigger(broken_invariant_names)
        if reason:
            engage(reason, source="ops_health_trigger")
            fired.append(reason)

    return fired


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m deep_isobar.ops.kill_switch",
        description="Deep Isobar kill switch — stop, inspect, or release live order placement.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="Print current state and exit.")
    group.add_argument("--engage", metavar="REASON", help="Engage the kill switch with REASON.")
    group.add_argument(
        "--release", action="store_true", help="Release the kill switch (requires --confirm)."
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Required alongside --release."
    )
    parser.add_argument(
        "--by", default=None,
        help="Who is engaging/releasing (defaults to $USER/$USERNAME).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    who = args.by or os.environ.get("USER") or os.environ.get("USERNAME") or "cli"

    if args.status:
        state = get_state()
        if state.engaged:
            print("ENGAGED — live order placement is refused")
            print(f"  reason:  {state.reason or state.detail or '(unknown)'}")
            print(f"  source:  {state.source or '(unknown)'}")
            print(f"  since:   {state.engaged_at_utc or '(unknown)'}")
        else:
            print("disengaged — live order placement is permitted")
        return 0

    if args.engage is not None:
        engage(reason=args.engage, source=f"human_cli:{who}")
        print(f"Kill switch ENGAGED by {who}: {args.engage}")
        return 0

    if args.release:
        if not args.confirm:
            print(
                "Refusing to release without --confirm — this is a deliberate "
                "action, not a default one.",
                file=sys.stderr,
            )
            return 1
        try:
            release(released_by=who, confirm=True)
        except ValueError as exc:
            print(f"Refused: {exc}", file=sys.stderr)
            return 1
        print(f"Kill switch RELEASED by {who}.")
        return 0

    return 1  # unreachable — the argument group is required


if __name__ == "__main__":
    raise SystemExit(main())
