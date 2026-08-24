"""Tests for deep_isobar.ops.kill_switch — the go-live safety build, Part 1.

Every test isolates the sentinel file and history log to tmp_path (never the
real ``data/KILL_SWITCH``) and stubs out Discord so nothing hits the network.
Covers: engage/release/is_engaged core behaviour, fail-closed semantics on
every kind of unreadable/corrupt state, automatic triggers firing exactly at
their threshold and not below it, the CLI, restart survival, and the three
independent call sites (submit_live_trade, run_preflight, session loop).
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from deep_isobar.ops import kill_switch


# ---------------------------------------------------------------------------
# Isolation fixture — every test gets its own sentinel file / history log
# under tmp_path, and a no-op, call-recording Discord stub.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_switch(monkeypatch, tmp_path):
    sentinel = tmp_path / "KILL_SWITCH"
    history = tmp_path / "logs" / "kill_switch_history.jsonl"
    monkeypatch.setattr(kill_switch, "_KILL_SWITCH_PATH", sentinel)
    monkeypatch.setattr(kill_switch, "_HISTORY_LOG_PATH", history)

    posted: list[dict] = []
    monkeypatch.setattr(
        kill_switch, "post_embed",
        lambda **kwargs: posted.append(kwargs),
    )
    return {"path": sentinel, "history": history, "posted": posted}


# ---------------------------------------------------------------------------
# Core state: disengaged by default, engage/release round trip
# ---------------------------------------------------------------------------


def test_disengaged_when_no_file(_isolated_switch):
    assert kill_switch.is_engaged() is False
    state = kill_switch.get_state()
    assert state.engaged is False


def test_engage_creates_sentinel_and_engages(_isolated_switch):
    kill_switch.engage(reason="manual test", source="unit_test")
    assert _isolated_switch["path"].exists()
    assert kill_switch.is_engaged() is True


def test_get_state_reports_reason_source_timestamp(_isolated_switch):
    kill_switch.engage(reason="daily loss too big", source="daily_loss_trigger")
    state = kill_switch.get_state()
    assert state.engaged is True
    assert state.reason == "daily loss too big"
    assert state.source == "daily_loss_trigger"
    assert state.engaged_at_utc is not None
    # Parseable ISO timestamp.
    datetime.fromisoformat(state.engaged_at_utc)


def test_engage_logs_critical_and_posts_discord(_isolated_switch, caplog):
    with caplog.at_level("CRITICAL"):
        kill_switch.engage(reason="panic button", source="human_cli:kaden")
    assert any("KILL SWITCH ENGAGED" in r.message for r in caplog.records)
    assert len(_isolated_switch["posted"]) == 1
    embed = _isolated_switch["posted"][0]
    assert "ENGAGED" in embed["title"]
    assert embed["color"] == kill_switch.COLOR_RED


def test_engage_appends_history_entry(_isolated_switch):
    kill_switch.engage(reason="r1", source="s1")
    lines = _isolated_switch["history"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "engage"
    assert entry["reason"] == "r1"
    assert entry["source"] == "s1"


def test_release_removes_file_and_disengages(_isolated_switch):
    kill_switch.engage(reason="r", source="s")
    assert kill_switch.is_engaged() is True
    kill_switch.release(released_by="kaden", confirm=True)
    assert kill_switch.is_engaged() is False
    assert not _isolated_switch["path"].exists()


def test_release_requires_confirm(_isolated_switch):
    kill_switch.engage(reason="r", source="s")
    with pytest.raises(ValueError, match="confirm"):
        kill_switch.release(released_by="kaden", confirm=False)
    # Refused — still engaged.
    assert kill_switch.is_engaged() is True


def test_release_requires_released_by(_isolated_switch):
    kill_switch.engage(reason="r", source="s")
    with pytest.raises(ValueError, match="released_by"):
        kill_switch.release(released_by="", confirm=True)
    assert kill_switch.is_engaged() is True


def test_release_is_noop_when_not_engaged(_isolated_switch):
    # No exception, no file created, nothing weird.
    kill_switch.release(released_by="kaden", confirm=True)
    assert kill_switch.is_engaged() is False


def test_release_logs_prior_reason_and_history(_isolated_switch):
    kill_switch.engage(reason="daily loss too big", source="daily_loss_trigger")
    kill_switch.release(released_by="kaden", confirm=True)

    lines = _isolated_switch["history"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    release_entry = json.loads(lines[1])
    assert release_entry["event"] == "release"
    assert release_entry["released_by"] == "kaden"
    assert release_entry["prior_reason"] == "daily loss too big"
    assert release_entry["prior_source"] == "daily_loss_trigger"

    # Discord release embed acknowledges the prior reason.
    release_embed = _isolated_switch["posted"][-1]
    assert "daily loss too big" in release_embed["fields"][0]["value"]


def test_engage_idempotent_across_repeated_calls(_isolated_switch):
    kill_switch.engage(reason="first", source="s1")
    kill_switch.engage(reason="second", source="s2")
    assert kill_switch.is_engaged() is True
    state = kill_switch.get_state()
    assert state.reason == "second"  # last write wins; still engaged either way


# ---------------------------------------------------------------------------
# Fail-closed semantics — the safety-critical core
# ---------------------------------------------------------------------------


def test_touched_empty_file_fails_closed(_isolated_switch):
    """A bare `touch data/KILL_SWITCH` from a terminal must engage the switch."""
    _isolated_switch["path"].parent.mkdir(parents=True, exist_ok=True)
    _isolated_switch["path"].touch()
    assert kill_switch.is_engaged() is True
    state = kill_switch.get_state()
    assert state.engaged is True
    assert "empty" in state.detail


def test_corrupt_json_fails_closed(_isolated_switch):
    _isolated_switch["path"].parent.mkdir(parents=True, exist_ok=True)
    _isolated_switch["path"].write_text("{not valid json", encoding="utf-8")
    assert kill_switch.is_engaged() is True
    state = kill_switch.get_state()
    assert state.engaged is True
    assert "corrupt" in state.detail


def test_json_array_instead_of_object_fails_closed(_isolated_switch):
    _isolated_switch["path"].parent.mkdir(parents=True, exist_ok=True)
    _isolated_switch["path"].write_text("[1, 2, 3]", encoding="utf-8")
    assert kill_switch.is_engaged() is True


def test_unreadable_file_fails_closed(monkeypatch, _isolated_switch):
    """File exists but read_text() raises (e.g. permissions) → still engaged."""
    _isolated_switch["path"].parent.mkdir(parents=True, exist_ok=True)
    _isolated_switch["path"].write_text('{"reason": "x", "source": "y"}', encoding="utf-8")

    target = _isolated_switch["path"]
    original_read_text = Path.read_text

    def _boom_read_text(self, *args, **kwargs):
        if self == target:
            raise OSError("permission denied (simulated)")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _boom_read_text)

    assert kill_switch.is_engaged() is True
    state = kill_switch.get_state()
    assert "unreadable" in state.detail


def test_existence_check_error_fails_closed(monkeypatch, _isolated_switch):
    """exists() itself raising (disk error) → engaged, never disengaged."""
    target = _isolated_switch["path"]
    original_exists = Path.exists

    def _boom_exists(self, *args, **kwargs):
        if self == target:
            raise OSError("disk error (simulated)")
        return original_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", _boom_exists)

    assert kill_switch.is_engaged() is True
    state = kill_switch.get_state()
    assert "path check failed" in state.detail


def test_state_survives_simulated_restart(monkeypatch, tmp_path):
    """No in-memory cache: a fresh module import still sees the file state."""
    sentinel = tmp_path / "KILL_SWITCH"
    history = tmp_path / "logs" / "kill_switch_history.jsonl"
    monkeypatch.setattr(kill_switch, "_KILL_SWITCH_PATH", sentinel)
    monkeypatch.setattr(kill_switch, "_HISTORY_LOG_PATH", history)
    monkeypatch.setattr(kill_switch, "post_embed", lambda **kw: None)

    kill_switch.engage(reason="pre-restart", source="unit_test")
    assert kill_switch.is_engaged() is True

    # Simulate a process restart: reload the module (re-runs top-level code,
    # discarding anything module-level), then re-point it at the same file
    # (a real restart would just re-import with the real, unchanged path).
    reloaded = importlib.reload(kill_switch)
    monkeypatch.setattr(reloaded, "_KILL_SWITCH_PATH", sentinel)
    monkeypatch.setattr(reloaded, "_HISTORY_LOG_PATH", history)

    assert reloaded.is_engaged() is True
    state = reloaded.get_state()
    assert state.reason == "pre-restart"

    # Leave the real module state clean for subsequent tests importing it.
    importlib.reload(kill_switch)


# ---------------------------------------------------------------------------
# Automatic triggers — pure functions, fire at threshold, not below
# ---------------------------------------------------------------------------


def test_daily_loss_trigger_fires_at_limit_not_below():
    assert kill_switch.check_daily_loss_trigger(-100.0, 100.0) is not None
    assert kill_switch.check_daily_loss_trigger(-99.99, 100.0) is None
    assert kill_switch.check_daily_loss_trigger(-150.0, 100.0) is not None


def test_daily_loss_trigger_ignores_profit():
    assert kill_switch.check_daily_loss_trigger(50.0, 100.0) is None


def test_daily_loss_trigger_disabled_when_limit_nonpositive():
    assert kill_switch.check_daily_loss_trigger(-1000.0, 0.0) is None
    assert kill_switch.check_daily_loss_trigger(-1000.0, -10.0) is None


def test_drawdown_trigger_fires_at_limit_not_below():
    # peak=500, bankroll=500 → 20% drawdown = $100.
    assert kill_switch.check_drawdown_trigger(400.0, 500.0, 500.0, 0.20) is not None
    assert kill_switch.check_drawdown_trigger(400.01, 500.0, 500.0, 0.20) is None
    assert kill_switch.check_drawdown_trigger(300.0, 500.0, 500.0, 0.20) is not None


def test_drawdown_trigger_ignores_new_peak():
    # current > peak (making new highs) → never triggers, even with huge pct.
    assert kill_switch.check_drawdown_trigger(600.0, 500.0, 500.0, 0.01) is None


def test_drawdown_trigger_disabled_when_bankroll_or_pct_nonpositive():
    assert kill_switch.check_drawdown_trigger(0.0, 500.0, 500.0, 0.0) is None
    assert kill_switch.check_drawdown_trigger(0.0, 500.0, 0.0, 0.20) is None


def test_consecutive_losses_trigger_fires_at_default_five_not_four():
    assert kill_switch.check_consecutive_losses_trigger(4) is None
    assert kill_switch.check_consecutive_losses_trigger(5) is not None
    assert kill_switch.check_consecutive_losses_trigger(6) is not None


def test_consecutive_losses_trigger_respects_custom_limit():
    assert kill_switch.check_consecutive_losses_trigger(2, max_consecutive_losses=3) is None
    assert kill_switch.check_consecutive_losses_trigger(3, max_consecutive_losses=3) is not None


def test_ops_health_trigger_fires_only_when_nonempty():
    assert kill_switch.check_ops_health_trigger([]) is None
    reason = kill_switch.check_ops_health_trigger(["stub_books", "params_age"])
    assert reason is not None
    assert "stub_books" in reason and "params_age" in reason


def test_reconciliation_trigger_fires_beyond_tolerance():
    assert kill_switch.check_reconciliation_trigger(100.0, 100.0, 1.0) is None
    assert kill_switch.check_reconciliation_trigger(100.0, 101.0, 1.0) is None  # == tolerance
    assert kill_switch.check_reconciliation_trigger(100.0, 101.5, 1.0) is not None


def test_evaluate_trading_triggers_engages_on_each_hit(monkeypatch, _isolated_switch):
    monkeypatch.setattr(
        kill_switch, "get_setting",
        lambda key, default=None: {
            "risk.position_sizing.bankroll_usd": 500.0,
            "risk.kill_switch.max_daily_loss_usd": 100.0,
            "risk.kill_switch.max_drawdown_pct": 0.20,
            "risk.kill_switch.max_consecutive_losses": 5,
        }.get(key, default),
    )

    fired = kill_switch.evaluate_trading_triggers(
        daily_realized_pnl_usd=-150.0,
        current_equity_usd=350.0,
        peak_equity_usd=500.0,
        consecutive_losses=6,
        broken_invariant_names=["stub_books"],
    )
    assert len(fired) == 4
    assert kill_switch.is_engaged() is True


def test_evaluate_trading_triggers_skips_unset_inputs(monkeypatch, _isolated_switch):
    monkeypatch.setattr(kill_switch, "get_setting", lambda key, default=None: default)
    fired = kill_switch.evaluate_trading_triggers()
    assert fired == []
    assert kill_switch.is_engaged() is False


def test_evaluate_trading_triggers_no_fire_below_thresholds(monkeypatch, _isolated_switch):
    monkeypatch.setattr(
        kill_switch, "get_setting",
        lambda key, default=None: {
            "risk.position_sizing.bankroll_usd": 500.0,
            "risk.kill_switch.max_daily_loss_usd": 100.0,
            "risk.kill_switch.max_drawdown_pct": 0.20,
            "risk.kill_switch.max_consecutive_losses": 5,
        }.get(key, default),
    )
    fired = kill_switch.evaluate_trading_triggers(
        daily_realized_pnl_usd=-10.0,
        current_equity_usd=490.0,
        peak_equity_usd=500.0,
        consecutive_losses=2,
        broken_invariant_names=[],
    )
    assert fired == []
    assert kill_switch.is_engaged() is False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_status_disengaged(_isolated_switch, capsys):
    rc = kill_switch.main(["--status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "disengaged" in out


def test_cli_engage_then_status(_isolated_switch, capsys):
    rc = kill_switch.main(["--engage", "test panic", "--by", "kaden"])
    assert rc == 0
    assert kill_switch.is_engaged() is True

    rc = kill_switch.main(["--status"])
    out = capsys.readouterr().out
    assert "ENGAGED" in out
    assert "test panic" in out


def test_cli_release_without_confirm_refuses(_isolated_switch, capsys):
    kill_switch.engage(reason="r", source="s")
    rc = kill_switch.main(["--release"])
    assert rc == 1
    assert kill_switch.is_engaged() is True


def test_cli_release_with_confirm_disengages(_isolated_switch, capsys):
    kill_switch.engage(reason="r", source="s")
    rc = kill_switch.main(["--release", "--confirm", "--by", "kaden"])
    assert rc == 0
    assert kill_switch.is_engaged() is False


def test_cli_engage_and_release_are_mutually_exclusive_flags():
    with pytest.raises(SystemExit):
        kill_switch.main(["--status", "--release"])


# ---------------------------------------------------------------------------
# Independent call-site wiring
# ---------------------------------------------------------------------------


def test_submit_live_trade_blocked_when_engaged(_isolated_switch):
    from unittest.mock import patch

    from deep_isobar.trading import trade_execution
    from deep_isobar.trading.trade_execution import submit_live_trade

    kill_switch.engage(reason="panic", source="unit_test")
    # Pass the paper/live gate so the engaged switch is what refuses, and
    # assert the exchange is never called.
    with patch.object(trade_execution, "get_setting",
                      lambda k, d=None: False if k == "runtime.paper_trade" else d), \
         patch.object(trade_execution.kalshi_client, "create_order") as create:
        with pytest.raises(kill_switch.KillSwitchEngagedError, match="panic"):
            submit_live_trade(
                market_source="Kalshi", contract_id="X-T90", side="BUY",
                quantity=1.0, price=0.5,
            )
    create.assert_not_called()


def test_submit_live_trade_paper_gate_refuses_when_disengaged(_isolated_switch):
    from deep_isobar.trading.trade_execution import (
        LiveTradingDisabledError,
        submit_live_trade,
    )

    # Disengaged switch, default config (runtime.paper_trade: true) — the
    # paper/live gate refuses before the kill-switch check is even reached.
    with pytest.raises(LiveTradingDisabledError, match="paper"):
        submit_live_trade(
            market_source="Kalshi", contract_id="X-T90", side="BUY",
            quantity=1.0, price=0.5,
        )


def _preflight_ok_kwargs(**overrides) -> dict:
    from deep_isobar.calibration.emos import EMOSParams

    fitted = datetime.now(timezone.utc)
    base = dict(
        city_name="Testville",
        effective_mean=90.0,
        effective_std=1.7,
        dist_source="EMOS",
        nbm_max_f=89.5,
        emos_params=EMOSParams(
            station_id="KTST", model_names=["GFS"],
            a0=0.0, a=[0.2], c=1.0, d=0.5,
            fitted_at_utc=fitted.isoformat(timespec="seconds"),
        ),
        n_contracts=10,
        market_is_live=True,
        hist_min_f=45.0,
        hist_max_f=98.0,
    )
    base.update(overrides)
    return base


def test_run_preflight_blocked_when_kill_switch_engaged(_isolated_switch):
    from deep_isobar.trading.preflight import run_preflight

    kill_switch.engage(reason="halt everything", source="unit_test")
    result = run_preflight(**_preflight_ok_kwargs())
    assert result.ok is False
    assert any("KILL SWITCH" in f for f in result.failures)


def test_run_preflight_kill_switch_gate_bypasses_disabled_config(monkeypatch, _isolated_switch):
    """risk.preflight.enabled: false must not be a way around the kill switch."""
    from deep_isobar.trading import preflight

    monkeypatch.setattr(
        preflight, "get_setting",
        lambda key, default=None: {"enabled": False} if key == "risk.preflight" else default,
    )
    kill_switch.engage(reason="halt everything", source="unit_test")
    result = preflight.run_preflight(**_preflight_ok_kwargs())
    assert result.ok is False
    assert any("KILL SWITCH" in f for f in result.failures)


def test_run_preflight_passes_when_disengaged(_isolated_switch):
    from deep_isobar.trading.preflight import run_preflight

    result = run_preflight(**_preflight_ok_kwargs())
    assert result.ok is True


def test_session_main_refuses_when_engaged(monkeypatch, _isolated_switch):
    from deep_isobar.research import paper_trade_session

    kill_switch.engage(reason="halt everything", source="unit_test")

    def _boom():
        raise AssertionError("get_city_universe must not be called when kill switch is engaged")

    monkeypatch.setattr(paper_trade_session, "get_city_universe", _boom)
    # Must return quietly, not raise — get_city_universe blowing up would
    # fail the test, proving the early-return happened before it was called.
    paper_trade_session.main(dry_run=True)


# ---------------------------------------------------------------------------
# ops/health.py integration
# ---------------------------------------------------------------------------


def test_ops_health_check_kill_switch_ok_when_disengaged(_isolated_switch):
    from deep_isobar.ops.health import OK, check_kill_switch

    check = check_kill_switch()
    assert check.status == OK


def test_ops_health_check_kill_switch_alarm_when_engaged(_isolated_switch):
    from deep_isobar.ops.health import ALARM, check_kill_switch

    kill_switch.engage(reason="halt", source="unit_test")
    check = check_kill_switch()
    assert check.status == ALARM
    assert "halt" in check.detail


def test_ops_health_main_engages_switch_on_broken_invariant(monkeypatch, _isolated_switch):
    from deep_isobar.ops import health

    broken = health.HealthCheck("stub_books", health.ALARM, "fake books detected")
    ok_kill_switch = health.HealthCheck("kill_switch", health.OK, "disengaged")
    monkeypatch.setattr(health, "run_health_checks", lambda: [broken, ok_kill_switch])
    monkeypatch.setattr(health, "post_alarm_embed", lambda checks: None)

    rc = health.main(["--no-discord"])
    assert rc == 1
    assert kill_switch.is_engaged() is True
    state = kill_switch.get_state()
    assert "stub_books" in state.reason
    assert state.source == "ops_health_trigger"


def test_ops_health_main_does_not_self_trigger_on_kill_switch_alone(monkeypatch, _isolated_switch):
    """An already-engaged switch showing as ALARM must not be its own trigger reason."""
    from deep_isobar.ops import health

    kill_switch.engage(reason="pre-existing", source="human_cli:kaden")
    only_kill_switch_alarm = health.HealthCheck(
        "kill_switch", health.ALARM, "ENGAGED — pre-existing"
    )
    monkeypatch.setattr(health, "run_health_checks", lambda: [only_kill_switch_alarm])
    monkeypatch.setattr(health, "post_alarm_embed", lambda checks: None)

    engage_calls = []
    monkeypatch.setattr(
        kill_switch, "engage",
        lambda reason, source: engage_calls.append((reason, source)),
    )

    health.main(["--no-discord"])
    assert engage_calls == []  # no new engage() call triggered by the switch itself


# ---------------------------------------------------------------------------
# settle_paper_trades.py trigger wiring
# ---------------------------------------------------------------------------


def test_settle_evaluates_daily_loss_trigger(monkeypatch, _isolated_switch):
    from deep_isobar.research import settle_paper_trades as sp

    monkeypatch.setattr(
        sp, "get_setting",
        lambda key, default=None: {
            "risk.position_sizing.bankroll_usd": 500.0,
        }.get(key, default),
    )
    monkeypatch.setattr(
        kill_switch, "get_setting",
        lambda key, default=None: {
            "risk.position_sizing.bankroll_usd": 500.0,
            "risk.kill_switch.max_daily_loss_usd": 20.0,
            "risk.kill_switch.max_drawdown_pct": 0.20,
            "risk.kill_switch.max_consecutive_losses": 5,
        }.get(key, default),
    )

    df = pd.DataFrame([
        {"date": "2026-08-01", "status": "LOSS", "realized_pnl": -30.0},
        {"date": "2026-08-02", "status": "LOSS", "realized_pnl": -25.0},
    ])
    sp._evaluate_kill_switch_triggers(df, session_pnl=-25.0)
    assert kill_switch.is_engaged() is True
    state = kill_switch.get_state()
    assert state.source == "daily_loss_trigger"


def test_settle_evaluates_consecutive_losses_from_history(monkeypatch, _isolated_switch):
    from deep_isobar.research import settle_paper_trades as sp

    monkeypatch.setattr(
        kill_switch, "get_setting",
        lambda key, default=None: {
            "risk.position_sizing.bankroll_usd": 500.0,
            "risk.kill_switch.max_daily_loss_usd": 10_000.0,   # won't fire
            "risk.kill_switch.max_drawdown_pct": 1.0,          # won't fire
            "risk.kill_switch.max_consecutive_losses": 3,
        }.get(key, default),
    )

    rows = [{"date": f"2026-08-0{i}", "status": "WIN", "realized_pnl": 1.0} for i in range(1, 3)]
    rows += [{"date": f"2026-08-0{i}", "status": "LOSS", "realized_pnl": -1.0} for i in range(3, 6)]
    df = pd.DataFrame(rows)

    sp._evaluate_kill_switch_triggers(df, session_pnl=-1.0)
    assert kill_switch.is_engaged() is True
    state = kill_switch.get_state()
    assert state.source == "consecutive_losses_trigger"


def test_settle_trigger_evaluation_never_raises_on_bad_data(monkeypatch, _isolated_switch, caplog):
    """A crash in trigger math must not take down settlement itself."""
    from deep_isobar.research import settle_paper_trades as sp

    df = pd.DataFrame([{"date": "2026-08-01", "status": "LOSS", "realized_pnl": "not-a-number"}])
    monkeypatch.setattr(sp, "get_setting", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    with caplog.at_level("ERROR"):
        sp._evaluate_kill_switch_triggers(df, session_pnl=-1.0)  # must not raise
