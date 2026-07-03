"""As-of data access for the Forecast Lab — lookahead is impossible here.

Every method requires ``as_of`` and filters to what was knowable at that
moment (see docs/DATA_STORE.md for the availability semantics of each
series).  Lab experiments must read data ONLY through this store; opening
the parquets directly is the loophole that turns a promising sweep result
into fiction.

    store = AsOfStore()
    fx  = store.forecasts("KMDW", as_of=date(2026, 6, 1))   # date <= as_of
    obs = store.settlements("KMDW", as_of=date(2026, 6, 1)) # date <  as_of
    books = store.market_books(as_of=datetime(..., tzinfo=utc))
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AsOfStore:
    """Read-only, as-of-time views over the Lab's three aligned series."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data = data_dir or (_PROJECT_ROOT / "data")

    # ── Forecast vintages ────────────────────────────────────────────────

    def forecasts(self, station_id: str, as_of: date) -> pd.DataFrame:
        """Member T+1 daily-max forecasts for target dates ``<= as_of``.

        The forecast for target date D was issued on D−1, so it is knowable
        on the morning of D itself.  ``actual_f`` is dropped — settlement
        truth comes only from :meth:`settlements`, with its stricter cutoff.
        """
        df = self._training_frame(station_id)
        df = df[df["date"] <= as_of]
        return df.drop(columns=["actual_f"], errors="ignore").reset_index(drop=True)

    def settlements(self, station_id: str, as_of: date) -> pd.DataFrame:
        """Observed settlement highs for dates strictly ``< as_of``.

        The high for day D posts the evening of D — on the morning of D it
        is not knowable, hence the strict inequality.
        """
        df = self._training_frame(station_id)
        df = df[(df["date"] < as_of) & df["actual_f"].notna()]
        return df[["date", "actual_f"]].reset_index(drop=True)

    def spread_history(self, station_id: str, as_of: date) -> pd.DataFrame:
        """Recorded live T+1 ensemble spread for target dates ``<= as_of``."""
        path = self._data / "emos_training" / f"{station_id}_t1_spread.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["date", "ens_spread_var"])
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[df["date"] <= as_of].reset_index(drop=True)

    # ── Market books ─────────────────────────────────────────────────────

    def market_books(
        self,
        as_of: datetime,
        city: str | None = None,
    ) -> pd.DataFrame:
        """Orderbook snapshots with ``snapshot_utc <= as_of``.

        Args:
            as_of: Timezone-aware UTC cutoff (naive datetimes are rejected —
                ambiguity is how lookahead sneaks in).
            city: Optional city filter.
        """
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware (UTC)")
        as_of = as_of.astimezone(timezone.utc)

        base = self._data / "market_history"
        if not base.exists():
            return pd.DataFrame()

        # Partition pruning: only read date= dirs at or before the cutoff.
        frames = []
        for day_dir in sorted(base.glob("date=*")):
            try:
                day = date.fromisoformat(day_dir.name.split("=", 1)[1])
            except ValueError:
                continue
            if day > as_of.date():
                continue
            for f in sorted(day_dir.glob("books_*.parquet")):
                try:
                    frames.append(pd.read_parquet(f))
                except Exception:  # noqa: BLE001
                    logger.exception("unreadable book file %s — skipped", f)
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        ts = pd.to_datetime(df["snapshot_utc"], utc=True)
        df = df[ts <= as_of]
        if city is not None:
            df = df[df["city"] == city]
        return df.reset_index(drop=True)

    # ── Internals ────────────────────────────────────────────────────────

    def _training_frame(self, station_id: str) -> pd.DataFrame:
        path = self._data / "emos_training" / f"{station_id}_t1_training.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No training parquet for {station_id} at {path} — "
                "run emos_training first (or point AsOfStore at the right data_dir)"
            )
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values("date")
