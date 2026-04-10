"""Deep Isobar Dashboard API — read-only FastAPI backend.

Serves data from CSV/parquet files on disk.  No live exchange connections.

Run with:
    uvicorn deep_isobar.dashboard.api:app --reload --port 8765
"""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAPER_TRADES_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"
_DAILY_LOG_CSV    = _PROJECT_ROOT / "data" / "paper_trades" / "daily_log.csv"
_BIAS_PROFILE_PQ  = _PROJECT_ROOT / "data" / "bias_profiles" / "KMDW_monthly_profile.parquet"

KALSHI_FEE = 0.07

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    found, missing = [], []
    for p in [_PAPER_TRADES_CSV, _DAILY_LOG_CSV, _BIAS_PROFILE_PQ]:
        (found if p.exists() else missing).append(p.name)
    print(f"[deep-isobar-api] data files found   : {found}")
    print(f"[deep-isobar-api] data files missing  : {missing}")
    yield


app = FastAPI(title="Deep Isobar Dashboard API", version="0.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_DTYPES = {
    "alpha": float,
    "model_prob": float,
    "market_prob": float,
    "ensemble_mean_f": float,
    "entry_price": float,
    "position_size": float,
    "threshold_f": float,
    "anomaly_penalty_f": float,
    "anomaly_adjusted_signal": float,
}


def _load_trades() -> pd.DataFrame:
    if not _PAPER_TRADES_CSV.exists():
        raise HTTPException(status_code=404, detail=f"paper_trades.csv not found at {_PAPER_TRADES_CSV}")
    df = pd.read_csv(_PAPER_TRADES_CSV, dtype=_CSV_DTYPES)
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    return df


def _load_daily_log() -> pd.DataFrame:
    if not _DAILY_LOG_CSV.exists():
        raise HTTPException(status_code=404, detail=f"daily_log.csv not found at {_DAILY_LOG_CSV}")
    df = pd.read_csv(_DAILY_LOG_CSV, dtype=_CSV_DTYPES)
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    return df


def _pnl_display(row: pd.Series) -> str:
    """Human-readable P&L string shown in trade list responses."""
    pnl = row.get("realized_pnl")
    if pd.isna(pnl):
        return "OPEN"
    return f"{float(pnl):+.4f}"


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a DataFrame row to a JSON-safe dict, normalising NaN → None."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        try:
            out[k] = None if pd.isna(v) else v
        except (TypeError, ValueError):
            # pd.isna raises on non-scalar containers — keep as-is
            out[k] = v
    out["pnl_display"] = _pnl_display(row)
    return out


def _compute_pnl(direction: str, entry_price: float, status: str, position_size: float) -> float:
    """Re-derive realized_pnl from direction + outcome status using Kalshi fee logic."""
    if status == "WIN":
        if direction == "BUY":
            return (1.0 - entry_price) * position_size * (1.0 - KALSHI_FEE)
        else:  # SELL
            return entry_price * position_size * (1.0 - KALSHI_FEE)
    else:  # LOSS
        if direction == "BUY":
            return -entry_price * position_size
        else:  # SELL
            return -(1.0 - entry_price) * position_size


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "generated_at": datetime.now(timezone.utc).isoformat()}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    df = _load_trades()

    settled = df[df["status"].isin(["WIN", "LOSS"])].copy()
    wins    = int((settled["status"] == "WIN").sum())
    losses  = int((settled["status"] == "LOSS").sum())
    n_settled = len(settled)

    win_rate = (wins / n_settled) if n_settled > 0 else None
    net_pnl  = float(settled["realized_pnl"].fillna(0.0).sum()) if n_settled > 0 else 0.0

    all_alpha = pd.to_numeric(df["alpha"], errors="coerce").dropna()
    avg_alpha = float(all_alpha.mean()) if not all_alpha.empty else 0.0

    best_pnl  = float(settled["realized_pnl"].fillna(0.0).max()) if n_settled > 0 else 0.0
    worst_pnl = float(settled["realized_pnl"].fillna(0.0).min()) if n_settled > 0 else 0.0

    # Daily Sharpe — requires at least 5 settled trades
    sharpe: float | None = None
    if n_settled >= 5 and "date" in settled.columns:
        daily = settled.groupby("date")["realized_pnl"].sum()
        std = float(daily.std())
        if std > 0:
            sharpe = float(daily.mean() / std * math.sqrt(252))

    return {
        "total_trades": len(df),
        "open_count":   int((df["status"] == "OPEN").sum()),
        "wins":         wins,
        "losses":       losses,
        "win_rate":     win_rate,
        "net_pnl":      net_pnl,
        "avg_alpha":    avg_alpha,
        "best_trade_pnl":  best_pnl,
        "worst_trade_pnl": worst_pnl,
        "sharpe":       sharpe,
    }


@app.get("/api/trades")
def trades(
    status: str = Query(default="ALL", description="OPEN|WIN|LOSS|ALL"),
    limit:  int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    valid_statuses = {"OPEN", "WIN", "LOSS", "ALL"}
    if status not in valid_statuses:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(valid_statuses)}")

    df = _load_trades()

    if status != "ALL":
        df = df[df["status"] == status]

    # Sort descending by date
    df = df.sort_values("date", ascending=False, kind="stable")

    total = len(df)
    page  = df.iloc[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "trades": [_row_to_dict(row) for _, row in page.iterrows()],
    }


@app.get("/api/daily_log")
def daily_log(
    date_param: str | None = Query(default=None, alias="date", description="YYYY-MM-DD"),
) -> dict[str, Any]:
    if date_param is not None:
        try:
            date.fromisoformat(date_param)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid date format: {date_param!r}. Use YYYY-MM-DD.")
        target = date_param
    else:
        target = str(date.today())

    df = _load_daily_log()
    rows = df[df["date"].astype(str) == target]

    return {
        "date": target,
        "count": len(rows),
        "rows": [_row_to_dict(row) for _, row in rows.iterrows()],
    }


@app.get("/api/pnl_curve")
def pnl_curve() -> list[dict[str, Any]]:
    df = _load_trades()
    settled = df[df["status"].isin(["WIN", "LOSS"])].copy()

    if settled.empty:
        return []

    settled["realized_pnl"] = settled["realized_pnl"].fillna(0.0)
    settled = settled.sort_values("date", ascending=True, kind="stable")

    # Aggregate to daily P&L (multiple trades per day possible)
    daily = (
        settled.groupby("date")
        .agg(
            daily_pnl=("realized_pnl", "sum"),
            win=("status", lambda s: (s == "WIN").any()),
        )
        .reset_index()
        .sort_values("date")
    )

    daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()

    return [
        {
            "date":           str(row["date"]),
            "cumulative_pnl": round(float(row["cumulative_pnl"]), 6),
            "daily_pnl":      round(float(row["daily_pnl"]), 6),
            "win":            bool(row["win"]),
        }
        for _, row in daily.iterrows()
    ]


@app.get("/api/alpha_distribution")
def alpha_distribution() -> list[dict[str, Any]]:
    df = _load_trades()
    alphas = pd.to_numeric(df["alpha"], errors="coerce").dropna()

    if alphas.empty:
        return []

    n_bins = 20
    counts, edges = pd.cut(alphas, bins=n_bins, retbins=True)  # type: ignore[assignment]
    bin_counts = counts.value_counts(sort=False)

    result = []
    for interval, count in bin_counts.items():
        lo = interval.left   # type: ignore[union-attr]
        hi = interval.right  # type: ignore[union-attr]
        midpoint = (lo + hi) / 2
        label = f"[{lo:.3f}, {hi:.3f})"
        result.append({
            "bucket_label": label,
            "count":        int(count),
            "is_positive":  bool(midpoint >= 0),
        })

    return result


_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@app.get("/api/bias_profile")
def bias_profile() -> dict[str, Any]:
    if not _BIAS_PROFILE_PQ.exists():
        return {"source": "not_available", "rows": []}

    try:
        df = pd.read_parquet(_BIAS_PROFILE_PQ)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read bias profile: {exc}")

    rows = []
    for _, row in df.iterrows():
        month_num = int(row["month"]) if "month" in row.index else 0
        month_name = _MONTH_NAMES[month_num - 1] if 1 <= month_num <= 12 else "Unknown"

        last_updated = None
        if "last_updated" in row.index and pd.notna(row["last_updated"]):
            last_updated = str(row["last_updated"])

        rows.append({
            "month":                int(month_num),
            "month_name":           month_name,
            "mean_bias_f":          float(row.get("mean_bias_f", 0.0)),
            "variance_multiplier":  float(row.get("variance_multiplier", 1.0)),
            "sample_count":         int(row.get("sample_count", 0)),
            "last_updated":         last_updated,
        })

    rows.sort(key=lambda r: r["month"])
    return {"source": "KMDW_monthly_profile.parquet", "rows": rows}


# ---------------------------------------------------------------------------
# PATCH /api/trades/{contract_ticker}
# ---------------------------------------------------------------------------

class TradeOverride(BaseModel):
    status: str
    settled_temp: float | None = None


@app.patch("/api/trades/{contract_ticker}")
def override_trade(contract_ticker: str, body: TradeOverride) -> dict[str, Any]:
    valid_statuses = {"WIN", "LOSS", "VOID"}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"status must be one of {sorted(valid_statuses)}",
        )

    if not _PAPER_TRADES_CSV.exists():
        raise HTTPException(status_code=404, detail="paper_trades.csv not found")

    df = pd.read_csv(_PAPER_TRADES_CSV, dtype=_CSV_DTYPES)
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")

    mask = df["contract_ticker"] == contract_ticker
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Contract {contract_ticker!r} not found")

    idx = df[mask].index[0]
    row = df.loc[idx]

    df.loc[idx, "status"] = body.status

    if body.settled_temp is not None:
        df.loc[idx, "settled_temp"] = body.settled_temp

    if body.status in ("WIN", "LOSS"):
        entry_price   = float(row["entry_price"])
        direction     = str(row["direction"])
        position_size = float(row["position_size"]) if pd.notna(row.get("position_size")) else 10.0
        pnl = _compute_pnl(direction, entry_price, body.status, position_size)
        df.loc[idx, "realized_pnl"] = round(pnl, 6)
    elif body.status == "VOID":
        df.loc[idx, "realized_pnl"] = 0.0

    df.to_csv(_PAPER_TRADES_CSV, index=False)

    updated_row = df.loc[idx]
    return _row_to_dict(updated_row)
