"""Numeric trade-flip gate — objective criteria for ``trade: false → true``.

Every flip so far was a judgment call, and the one that skipped the process
(Los Angeles, 2026-08-05, flipped on book depth + calibration) went 1W/15L
before being un-flipped.  This module turns the flip decision into a
reproducible report: for every active, non-trading city it scores

- **Calibration** (reusing the daily scorecard's ``station_calibration``,
  so the gate and the nightly report can never disagree): enough scored
  days, CRPS under the ceiling, EMOS MAE beating NBM.
- **Liquidity** (from the orderbook collector's archive — real books only;
  the stub-book contamination era ended 2026-07-12): median top-of-book
  spread, share of two-sided quotes, median 24h volume across the ladder.
- **Track record**: a city with a real settled paper history must not be
  re-flipped past it — with ``min_track_trades``+ settled trades in the
  last 60 days, realized P&L must be positive.  This is the criterion that
  keeps LA out until a marine-layer fix wins the holdout race.

Thresholds live in ``config/settings.yaml`` under ``flip_gate:``.

Usage::

    python -m deep_isobar.research.flip_gate            # report all candidates
    python -m deep_isobar.research.flip_gate --city Miami

The tool only reports — flipping ``trade:`` in cities.yaml stays a
deliberate human edit, like releasing the kill switch.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from deep_isobar.config import get_setting
from deep_isobar.core.types import CityProfile
from deep_isobar.data.city_universe import get_city_universe
from deep_isobar.research.daily_scorecard import (
    load_settled_trades,
    station_calibration,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_HISTORY_DIR = _PROJECT_ROOT / "data" / "market_history"

_BOOK_COLUMNS = [
    "snapshot_utc", "series", "metric", "best_bid", "best_ask", "volume_24h",
]


@dataclass
class Criterion:
    """One pass/fail check with the measured value behind it."""

    name: str
    ok: bool
    detail: str


@dataclass
class GateResult:
    """Gate outcome for one city."""

    city: str
    station_id: str
    series: str
    criteria: list[Criterion] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.criteria) and all(c.ok for c in self.criteria)


def load_book_window(
    history_dir: Path | None = None,
    days: int = 14,
    asof: date | None = None,
) -> pd.DataFrame:
    """Read every book snapshot in the window ONCE (all series together).

    ~90 files/day; callers slice the returned frame per series rather than
    re-reading the archive per city.
    """
    root = history_dir or _HISTORY_DIR
    today = asof or date.today()
    frames: list[pd.DataFrame] = []
    for offset in range(days):
        day_dir = root / f"date={today - timedelta(days=offset)}"
        if not day_dir.is_dir():
            continue
        for book in sorted(day_dir.glob("books_*.parquet")):
            try:
                frames.append(pd.read_parquet(book, columns=_BOOK_COLUMNS))
            except Exception:  # noqa: BLE001 — one bad file must not kill the report
                logger.warning("unreadable book snapshot skipped: %s", book)
    if not frames:
        return pd.DataFrame(columns=_BOOK_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def liquidity_stats(
    series: str,
    history_dir: Path | None = None,
    days: int = 14,
    asof: date | None = None,
    metric: str = "high",
    books: pd.DataFrame | None = None,
) -> dict | None:
    """Aggregate top-of-book liquidity for one series from the book archive.

    Pass *books* (from :func:`load_book_window`) to reuse one archive read
    across many cities.  Returns ``None`` when no snapshots exist in the
    window (collector gap or freshly onboarded city).
    """
    if books is None:
        books = load_book_window(history_dir, days, asof)
    rows = books[(books["series"] == series) & (books["metric"] == metric)]
    if rows.empty:
        return None
    bid = pd.to_numeric(rows["best_bid"], errors="coerce")
    ask = pd.to_numeric(rows["best_ask"], errors="coerce")
    two_sided = (bid > 0) & (ask > 0) & (ask < 1)
    spreads = (ask - bid)[two_sided]

    volume = pd.to_numeric(rows["volume_24h"], errors="coerce").fillna(0.0)
    per_snapshot_volume = volume.groupby(rows["snapshot_utc"]).sum()

    return {
        "n_snapshots": int(rows["snapshot_utc"].nunique()),
        "n_quotes": int(len(rows)),
        "two_sided_frac": float(two_sided.mean()),
        "median_spread": float(spreads.median()) if len(spreads) else float("nan"),
        "median_volume_24h": float(per_snapshot_volume.median()),
    }


def track_record(city_name: str, days: int = 60, asof: date | None = None) -> dict:
    """Settled paper-trade count and P&L for one city over the last *days*."""
    trades = load_settled_trades()
    if trades.empty:
        return {"n": 0, "pnl": 0.0}
    today = asof or date.today()
    cutoff = today - timedelta(days=days)
    dates = pd.to_datetime(trades["date"]).dt.date
    w = trades[(trades["city"] == city_name) & (dates >= cutoff)]
    return {
        "n": int(len(w)),
        "pnl": float(pd.to_numeric(w["realized_pnl"], errors="coerce").fillna(0.0).sum()),
    }


def evaluate_city(
    city: CityProfile,
    calib: dict | None,
    liq: dict | None,
    track: dict,
    cfg: dict,
) -> GateResult:
    """Score one city against the configured gate — pure, fully testable."""
    result = GateResult(
        city=city.city,
        station_id=city.station_id,
        series=getattr(city, "kalshi_series", "") or "",
    )
    add = result.criteria.append

    min_days = int(cfg.get("min_calibration_days", 21))
    max_crps = float(cfg.get("max_crps_f", 1.0))
    if calib is None:
        add(Criterion("calibration", False, "no params or too few scored days"))
    else:
        add(Criterion(
            "calibration_days", calib["n_days"] >= min_days,
            f"{calib['n_days']}d scored (need {min_days})",
        ))
        add(Criterion(
            "crps", calib["crps"] <= max_crps,
            f"CRPS {calib['crps']:.3f} (max {max_crps:.2f})",
        ))
        if cfg.get("require_beats_nbm", True):
            nbm = calib.get("nbm_mae")
            has_nbm = nbm is not None and nbm == nbm  # NaN-safe
            add(Criterion(
                "beats_nbm", bool(has_nbm and calib["mae"] < nbm),
                f"MAE {calib['mae']:.2f} vs NBM {nbm:.2f}" if has_nbm
                else "no NBM benchmark rows",
            ))

    max_spread = float(cfg.get("max_median_spread", 0.06))
    min_two_sided = float(cfg.get("min_two_sided_frac", 0.75))
    min_volume = float(cfg.get("min_series_volume_24h", 100.0))
    if liq is None:
        add(Criterion("liquidity", False, "no book snapshots in window"))
    else:
        spread = liq["median_spread"]
        add(Criterion(
            "spread", spread == spread and spread <= max_spread,
            f"median spread {spread:.2f} (max {max_spread:.2f})",
        ))
        add(Criterion(
            "two_sided", liq["two_sided_frac"] >= min_two_sided,
            f"{liq['two_sided_frac']:.0%} two-sided (need {min_two_sided:.0%})",
        ))
        add(Criterion(
            "volume", liq["median_volume_24h"] >= min_volume,
            f"median 24h vol {liq['median_volume_24h']:.0f} (need {min_volume:.0f})",
        ))

    min_track = int(cfg.get("min_track_trades", 10))
    if track["n"] >= min_track:
        add(Criterion(
            "track_record", track["pnl"] > 0.0,
            f"{track['n']} settled, P&L {track['pnl']:+.2f} — a losing paper "
            "book blocks the flip" if track["pnl"] <= 0.0
            else f"{track['n']} settled, P&L {track['pnl']:+.2f}",
        ))
    else:
        add(Criterion(
            "track_record", True,
            f"only {track['n']} settled trades — no live evidence against",
        ))

    return result


def run_gate(
    only_city: str | None = None,
    asof: date | None = None,
) -> list[GateResult]:
    """Evaluate every active, non-trading city (or one named city)."""
    cfg = get_setting("flip_gate", default={}) or {}
    window_days = int(cfg.get("window_days", 30))
    liquidity_days = int(cfg.get("liquidity_days", 14))

    results: list[GateResult] = []
    for city in get_city_universe():
        if not city.active:
            continue
        if only_city is not None:
            if city.city.lower() != only_city.lower():
                continue
        elif getattr(city, "trade", False):
            continue  # already trading — the gate is for candidates
        calib = station_calibration(city, window_days=window_days, asof=asof)
        liq = liquidity_stats(
            getattr(city, "kalshi_series", "") or "",
            days=liquidity_days,
            asof=asof,
        )
        results.append(evaluate_city(
            city, calib, liq, track_record(city.city, asof=asof), cfg,
        ))
    return results


def render_report(results: list[GateResult]) -> str:
    """Markdown report: verdict per city, one line per criterion."""
    lines = ["# Trade-flip gate report", ""]
    passed = [r for r in results if r.passed]
    lines.append(
        f"**{len(passed)} of {len(results)} candidates pass:** "
        + (", ".join(r.city for r in passed) if passed else "none")
    )
    lines.append("")
    for r in sorted(results, key=lambda r: (not r.passed, r.city)):
        verdict = "✅ PASS" if r.passed else "❌ FAIL"
        lines.append(f"## {r.city} ({r.station_id}, {r.series}) — {verdict}")
        for c in r.criteria:
            mark = "✅" if c.ok else "❌"
            lines.append(f"- {mark} `{c.name}` — {c.detail}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city", help="Evaluate one city (even if already trading).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    results = run_gate(only_city=args.city)
    if not results:
        print("No matching candidate cities.")
        return 1
    print(render_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
