"""Deep Isobar Dashboard API — FastAPI backend.

Serves data from CSV/parquet files on disk plus SQLite-backed user auth.

Run with:
    uvicorn deep_isobar.dashboard.api:app --reload --port 8765
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import bcrypt as _bcrypt_lib
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from deep_isobar.dashboard.audit import fetch_audit, init_audit_table, record_audit
from deep_isobar.dashboard.exports import xlsx_response

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT     = Path(__file__).resolve().parents[3]
_PAPER_TRADES_CSV = _PROJECT_ROOT / "data" / "paper_trades" / "paper_trades.csv"
_DAILY_LOG_CSV    = _PROJECT_ROOT / "data" / "paper_trades" / "daily_log.csv"
_BIAS_PROFILE_PQ  = _PROJECT_ROOT / "data" / "bias_profiles" / "KMDW_monthly_profile.parquet"
_DB_PATH          = _PROJECT_ROOT / "data" / "users.db"
_ALERTS_JSON      = _PROJECT_ROOT / "data" / "alerts.json"
_SETTINGS_YAML    = _PROJECT_ROOT / "config" / "settings.yaml"
_ENV_PATH         = _PROJECT_ROOT / ".env"

KALSHI_FEE = 0.07

# ---------------------------------------------------------------------------
# Load env + auth config
# ---------------------------------------------------------------------------

load_dotenv(_ENV_PATH)

_JWT_SECRET    = os.getenv("JWT_SECRET", "dev-secret-do-not-use-in-production")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_H  = 24
_security      = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    return _bcrypt_lib.hashpw(password.encode(), _bcrypt_lib.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return _bcrypt_lib.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create users table and seed admin if none exists."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_db()
    init_audit_table(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id                  TEXT PRIMARY KEY,
            username            TEXT UNIQUE NOT NULL,
            display_name        TEXT NOT NULL,
            email               TEXT DEFAULT '',
            hashed_password     TEXT NOT NULL,
            role                TEXT NOT NULL,
            is_active           INTEGER DEFAULT 1,
            must_change_password INTEGER DEFAULT 0,
            created_at          TEXT NOT NULL,
            last_login          TEXT DEFAULT NULL
        )
    """)
    conn.commit()

    existing_admin = conn.execute(
        "SELECT id FROM users WHERE role = 'admin' LIMIT 1"
    ).fetchone()

    if not existing_admin:
        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_password = os.getenv("ADMIN_PASSWORD", "changeme_on_first_run")
        conn.execute(
            """INSERT INTO users
               (id, username, display_name, email, hashed_password, role,
                is_active, must_change_password, created_at)
               VALUES (?, ?, ?, '', ?, 'admin', 1, 0, ?)""",
            (
                str(uuid.uuid4()),
                admin_username,
                admin_username.upper(),
                _hash_password(admin_password),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        print(f"[deep-isobar-api] Admin user seeded from .env  (username: {admin_username})")

    conn.close()


def _user_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d.pop("hashed_password", None)
    d["is_active"] = bool(d.get("is_active", 1))
    d["must_change_password"] = bool(d.get("must_change_password", 0))
    return d

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _create_token(username: str, role: str, must_change_password: bool, display_name: str = "") -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=_JWT_EXPIRE_H)
    payload = {
        "sub":                  username,
        "role":                 role,
        "display_name":         display_name or username,
        "must_change_password": must_change_password,
        "exp":                  expire,
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_security),
) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = credentials.credentials
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        username: str = payload.get("sub", "")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if row is None or not row["is_active"]:
        raise HTTPException(status_code=401, detail="User not found or deactivated")

    user = dict(row)
    user["must_change_password"] = bool(user.get("must_change_password", 0))
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def require_investor_or_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") not in ("admin", "investor"):
        raise HTTPException(status_code=403, detail="Investor or admin access required")
    return user

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    _init_db()
    found, missing = [], []
    for p in [_PAPER_TRADES_CSV, _DAILY_LOG_CSV, _BIAS_PROFILE_PQ]:
        (found if p.exists() else missing).append(p.name)
    print(f"[deep-isobar-api] data files found   : {found}")
    print(f"[deep-isobar-api] data files missing  : {missing}")
    yield


app = FastAPI(title="Deep Isobar Dashboard API", version="0.2.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# CSV helpers (unchanged from v0.1)
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
}


def _load_trades() -> pd.DataFrame:
    if not _PAPER_TRADES_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=f"paper_trades.csv not found at {_PAPER_TRADES_CSV}",
        )
    df = pd.read_csv(_PAPER_TRADES_CSV, dtype=_CSV_DTYPES)
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    return df


def _load_daily_log() -> pd.DataFrame:
    if not _DAILY_LOG_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=f"daily_log.csv not found at {_DAILY_LOG_CSV}",
        )
    df = pd.read_csv(_DAILY_LOG_CSV, dtype=_CSV_DTYPES)
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    return df


def _pnl_display(row: pd.Series) -> str:
    pnl = row.get("realized_pnl")
    if pd.isna(pnl):
        return "OPEN"
    return f"{float(pnl):+.4f}"


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        try:
            out[k] = None if pd.isna(v) else v
        except (TypeError, ValueError):
            out[k] = v
    out["pnl_display"] = _pnl_display(row)
    out["spread_rank"] = int(out.get("spread_rank") or 1)
    out["spread_total_contracts"] = int(out.get("spread_total_contracts") or 1)
    return out


def _compute_pnl(
    direction: str, entry_price: float, status: str, position_size: float
) -> float:
    if status == "WIN":
        if direction == "BUY":
            return (1.0 - entry_price) * position_size * (1.0 - KALSHI_FEE)
        else:
            return entry_price * position_size * (1.0 - KALSHI_FEE)
    else:  # LOSS
        if direction == "BUY":
            return -entry_price * position_size
        else:
            return -(1.0 - entry_price) * position_size

# ---------------------------------------------------------------------------
# Alerts helpers
# ---------------------------------------------------------------------------

def _load_alerts() -> list[dict]:
    if not _ALERTS_JSON.exists():
        return []
    try:
        return json.loads(_ALERTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_alerts(alerts: list[dict]) -> None:
    _ALERTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    _ALERTS_JSON.write_text(
        json.dumps(alerts, indent=2, default=str), encoding="utf-8"
    )

# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

def _load_yaml() -> dict:
    if not _SETTINGS_YAML.exists():
        return {}
    with open(_SETTINGS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(data: dict) -> None:
    with open(_SETTINGS_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def _read_env_vars() -> dict[str, str]:
    """Read .env file as key-value pairs, ignoring comments/blanks."""
    result: dict[str, str] = {}
    if not _ENV_PATH.exists():
        return result
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def _write_env_key(key: str, value: str) -> None:
    """Upsert a key in the .env file."""
    if not _ENV_PATH.exists():
        _ENV_PATH.write_text(f"{key}={value}\n", encoding="utf-8")
        return
    lines = _ENV_PATH.read_text(encoding="utf-8").splitlines()
    new_lines, found = [], False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#") and stripped.startswith(key + "="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    _ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Investor computation helpers
# ---------------------------------------------------------------------------

_INITIAL_CAPITAL = 100.0
_MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def _compute_sharpe(pnl_series: pd.Series) -> Optional[float]:
    """Annualised daily Sharpe from a daily P&L series.  None if insufficient data."""
    if len(pnl_series) < 2:
        return None
    std = float(pnl_series.std())
    if std == 0:
        return None
    return float(pnl_series.mean() / std * math.sqrt(252))


def _period_stats(group: pd.DataFrame, date_col: str = "date") -> dict:
    """Aggregate stats for a group of settled trades."""
    n = len(group)
    wins = int((group["status"] == "WIN").sum())
    net = float(group["realized_pnl"].fillna(0.0).sum())
    daily = group.groupby(group[date_col].astype(str))["realized_pnl"].sum()
    sharpe = _compute_sharpe(daily) if n >= 2 else None
    return {
        "net_return_usd": round(net, 4),
        "return_pct":     round(net / _INITIAL_CAPITAL * 100, 4),
        "trades":         n,
        "wins":           wins,
        "win_rate":       round(wins / n, 4) if n > 0 else 0.0,
        "sharpe":         round(sharpe, 4) if sharpe is not None else None,
    }

# ---------------------------------------------------------------------------
# UNAUTHENTICATED ENDPOINT
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "generated_at": datetime.now(timezone.utc).isoformat()}

# ---------------------------------------------------------------------------
# AUTH ENDPOINTS
# ---------------------------------------------------------------------------

class LoginBody(BaseModel):
    username: str
    password: str


class ChangePasswordBody(BaseModel):
    current_password: str
    new_password: str


@app.post("/api/auth/login")
def login(body: LoginBody, request: Request) -> dict:
    client_ip = request.client.host if request.client else ""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (body.username,)
    ).fetchone()

    if row is None or not _verify_password(body.password, row["hashed_password"]):
        record_audit(conn, body.username, "login.failure", ip=client_ip)
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not row["is_active"]:
        record_audit(conn, body.username, "login.deactivated", ip=client_ip)
        conn.close()
        raise HTTPException(status_code=403, detail="Account deactivated")

    # Update last_login
    conn.execute(
        "UPDATE users SET last_login = ? WHERE username = ?",
        (datetime.now(timezone.utc).isoformat(), body.username),
    )
    conn.commit()
    record_audit(conn, body.username, "login.success", ip=client_ip)
    conn.close()

    must_change = bool(row["must_change_password"])
    token = _create_token(body.username, row["role"], must_change, row["display_name"])
    return {
        "access_token":          token,
        "token_type":            "bearer",
        "role":                  row["role"],
        "username":              row["username"],
        "display_name":          row["display_name"],
        "must_change_password":  must_change,
    }


@app.get("/api/auth/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return {
        "username":             user["username"],
        "display_name":         user["display_name"],
        "role":                 user["role"],
        "email":                user.get("email", ""),
        "must_change_password": bool(user.get("must_change_password", 0)),
        "last_login":           user.get("last_login"),
    }


@app.post("/api/auth/change-password")
def change_password(
    body: ChangePasswordBody,
    user: dict = Depends(get_current_user),
) -> dict:
    if not _verify_password(body.current_password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    conn = _get_db()
    conn.execute(
        "UPDATE users SET hashed_password = ?, must_change_password = 0 WHERE id = ?",
        (_hash_password(body.new_password), user["id"]),
    )
    conn.commit()
    conn.close()
    return {"success": True}

# ---------------------------------------------------------------------------
# USER MANAGEMENT (admin only)
# ---------------------------------------------------------------------------

class CreateUserBody(BaseModel):
    username: str
    display_name: str
    email: str = ""
    temp_password: str


class PatchUserBody(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None
    temp_password: Optional[str] = None


@app.get("/api/users")
def list_users(admin: dict = Depends(require_admin)) -> list[dict]:
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM users ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [_user_row_to_dict(r) for r in rows]


@app.post("/api/users", status_code=201)
def create_user(body: CreateUserBody, admin: dict = Depends(require_admin)) -> dict:
    if not re.match(r"^[a-zA-Z0-9_]{3,32}$", body.username):
        raise HTTPException(
            status_code=422,
            detail="Username must be 3–32 alphanumeric/underscore characters",
        )
    if len(body.temp_password) < 6:
        raise HTTPException(status_code=422, detail="Temp password must be at least 6 characters")

    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO users
               (id, username, display_name, email, hashed_password, role,
                is_active, must_change_password, created_at)
               VALUES (?, ?, ?, ?, ?, 'investor', 1, 1, ?)""",
            (
                user_id,
                body.username,
                body.display_name,
                body.email,
                _hash_password(body.temp_password),
                now,
            ),
        )
        conn.commit()
        record_audit(conn, admin["username"], "users.create",
                     detail=f"created {body.username!r} (investor)")
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Username already exists")
    finally:
        conn.close()
    return _user_row_to_dict(row)


@app.patch("/api/users/{user_id}")
def patch_user(
    user_id: str,
    body: PatchUserBody,
    admin: dict = Depends(require_admin),
) -> dict:
    conn = _get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    # Block changing admin's is_active
    if row["role"] == "admin" and body.is_active is not None:
        conn.close()
        raise HTTPException(status_code=403, detail="Cannot modify admin account status")

    updates, params = [], []
    if body.display_name is not None:
        updates.append("display_name = ?")
        params.append(body.display_name)
    if body.email is not None:
        updates.append("email = ?")
        params.append(body.email)
    if body.is_active is not None and row["role"] != "admin":
        updates.append("is_active = ?")
        params.append(1 if body.is_active else 0)
    if body.temp_password is not None:
        updates.append("hashed_password = ?")
        params.append(_hash_password(body.temp_password))
        updates.append("must_change_password = 1")

    if updates:
        params.append(user_id)
        conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params
        )
        conn.commit()
        record_audit(
            conn, admin["username"], "users.patch",
            detail=f"{row['username']!r}: {', '.join(u.split(' = ')[0] for u in updates)}",
        )

    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return _user_row_to_dict(row)


@app.delete("/api/users/{user_id}")
def delete_user(user_id: str, admin: dict = Depends(require_admin)) -> dict:
    conn = _get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    if row["role"] == "admin":
        conn.close()
        raise HTTPException(status_code=403, detail="Cannot delete admin account")
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    record_audit(conn, admin["username"], "users.delete",
                 detail=f"deleted {row['username']!r}")
    conn.close()
    return {"deleted": True}

# ---------------------------------------------------------------------------
# INVESTOR ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/api/investor/summary")
def investor_summary(user: dict = Depends(require_investor_or_admin)) -> dict:
    if not _PAPER_TRADES_CSV.exists():
        return {
            "total_pnl": 0.0, "open_positions": 0, "settled_trades": 0,
            "wins": 0, "win_rate": 0.0, "avg_alpha": 0.0,
            "first_trade_date": None, "days_elapsed": 1,
            "initial_capital": _INITIAL_CAPITAL, "annualized_roi_pct": 0.0,
            "sharpe_estimate": None, "kalshi_balance": None,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    df = _load_trades()
    settled  = df[df["status"].isin(["WIN", "LOSS"])].copy()
    wins     = int((settled["status"] == "WIN").sum())
    n_settled = len(settled)
    total_pnl = float(settled["realized_pnl"].fillna(0.0).sum()) if n_settled > 0 else 0.0

    non_void  = df[df["status"] != "VOID"]
    avg_alpha = float(
        pd.to_numeric(non_void["alpha"], errors="coerce").dropna().mean()
    ) if not non_void.empty else 0.0

    dates = pd.to_datetime(df["date"].astype(str), errors="coerce").dropna()
    first_trade_date = str(dates.min().date()) if not dates.empty else None
    if first_trade_date:
        days_elapsed = max(1, (date.today() - date.fromisoformat(first_trade_date)).days)
    else:
        days_elapsed = 1

    annualized_roi = (total_pnl / _INITIAL_CAPITAL) * (365 / days_elapsed) * 100

    # Sharpe from daily settled P&L
    sharpe = None
    if n_settled >= 2:
        daily = settled.groupby(settled["date"].astype(str))["realized_pnl"].sum()
        sharpe = _compute_sharpe(daily)

    # Kalshi balance
    kalshi_balance = None
    try:
        from deep_isobar.market.kalshi_client import get_balance
        result = get_balance()
        if result and not result.get("error"):
            kalshi_balance = result.get("balance_usd")
    except Exception:
        pass

    return {
        "total_pnl":          round(total_pnl, 4),
        "open_positions":     int((df["status"] == "OPEN").sum()),
        "settled_trades":     n_settled,
        "wins":               wins,
        "win_rate":           round(wins / n_settled, 4) if n_settled > 0 else 0.0,
        "avg_alpha":          round(avg_alpha, 6),
        "first_trade_date":   first_trade_date,
        "days_elapsed":       days_elapsed,
        "initial_capital":    _INITIAL_CAPITAL,
        "annualized_roi_pct": round(annualized_roi, 4),
        "sharpe_estimate":    round(sharpe, 4) if sharpe is not None else None,
        "kalshi_balance":     kalshi_balance,
        "last_updated":       datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/investor/performance")
def investor_performance(user: dict = Depends(require_investor_or_admin)) -> dict:
    if not _PAPER_TRADES_CSV.exists():
        return {"monthly": [], "yearly": []}

    df = _load_trades()
    settled = df[df["status"].isin(["WIN", "LOSS"])].copy()
    settled["realized_pnl"] = settled["realized_pnl"].fillna(0.0)
    settled["date_str"]     = settled["date"].astype(str)
    settled["year"]         = settled["date_str"].str[:4]
    settled["year_month"]   = settled["date_str"].str[:7]

    monthly_rows: list[dict] = []
    for ym, grp in settled.groupby("year_month", sort=True):
        year_s, month_s = str(ym).split("-")
        month_name = _MONTH_ABBR[int(month_s) - 1]
        stats = _period_stats(grp, date_col="date_str")
        monthly_rows.append({"period": f"{month_name} {year_s}", "year_month": ym, **stats})

    yearly_rows: list[dict] = []
    for year, grp in settled.groupby("year", sort=True):
        stats = _period_stats(grp, date_col="date_str")
        yearly_rows.append({"period": str(year), "year_month": str(year), **stats})

    return {"monthly": monthly_rows, "yearly": yearly_rows}

# ---------------------------------------------------------------------------
# ALERTS ENDPOINTS
# ---------------------------------------------------------------------------

class CreateAlertBody(BaseModel):
    city: str
    station: str
    trigger_type: str
    threshold: float
    sensitivity: str


class PatchAlertBody(BaseModel):
    status: Optional[str] = None
    threshold: Optional[float] = None
    sensitivity: Optional[str] = None


@app.get("/api/alerts")
def list_alerts(user: dict = Depends(get_current_user)) -> list[dict]:
    return _load_alerts()


@app.post("/api/alerts", status_code=201)
def create_alert(body: CreateAlertBody, admin: dict = Depends(require_admin)) -> dict:
    valid_trigger  = {"alpha_gt", "alpha_lt", "drift_breach", "bias_warning"}
    valid_sens     = {"immediate", "delayed"}
    if body.trigger_type not in valid_trigger:
        raise HTTPException(status_code=422, detail=f"trigger_type must be one of {sorted(valid_trigger)}")
    if body.sensitivity not in valid_sens:
        raise HTTPException(status_code=422, detail=f"sensitivity must be one of {sorted(valid_sens)}")

    alert = {
        "id":            str(uuid.uuid4()),
        "city":          body.city,
        "station":       body.station,
        "trigger_type":  body.trigger_type,
        "threshold":     body.threshold,
        "sensitivity":   body.sensitivity,
        "status":        "active",
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "last_triggered": None,
        "trigger_count":  0,
    }
    alerts = _load_alerts()
    alerts.append(alert)
    _save_alerts(alerts)
    return alert


@app.patch("/api/alerts/{alert_id}")
def patch_alert(
    alert_id: str, body: PatchAlertBody, admin: dict = Depends(require_admin)
) -> dict:
    alerts = _load_alerts()
    for i, a in enumerate(alerts):
        if a["id"] == alert_id:
            if body.status    is not None: alerts[i]["status"]      = body.status
            if body.threshold is not None: alerts[i]["threshold"]   = body.threshold
            if body.sensitivity is not None: alerts[i]["sensitivity"] = body.sensitivity
            _save_alerts(alerts)
            return alerts[i]
    raise HTTPException(status_code=404, detail="Alert not found")


@app.delete("/api/alerts/{alert_id}")
def delete_alert(alert_id: str, admin: dict = Depends(require_admin)) -> dict:
    alerts = _load_alerts()
    new_alerts = [a for a in alerts if a["id"] != alert_id]
    if len(new_alerts) == len(alerts):
        raise HTTPException(status_code=404, detail="Alert not found")
    _save_alerts(new_alerts)
    return {"deleted": True}

# ---------------------------------------------------------------------------
# SETTINGS ENDPOINTS
# ---------------------------------------------------------------------------

class PatchSettingsBody(BaseModel):
    alpha_threshold: Optional[float] = None
    lead_decay_halflife_hours: Optional[float] = None
    discord_webhook_url: Optional[str] = None


def _build_settings_response() -> dict:
    cfg = _load_yaml()
    env_vars = _read_env_vars()
    discord_url = env_vars.get("DISCORD_WEBHOOK_URL", "")
    kalshi_key  = env_vars.get("KALSHI_API_KEY_ID", "") or env_vars.get("KALSHI_API_KEY", "")
    return {
        "yaml":                     cfg,
        "alpha_threshold":          (cfg.get("risk") or {}).get("alpha_threshold"),
        "lead_decay_halflife_hours": (cfg.get("risk") or {}).get("lead_decay_halflife_hours"),
        "discord_webhook_configured": bool(discord_url),
        "kalshi_api_configured":    bool(kalshi_key),
        "kalshi_api_key_last4":     kalshi_key[-4:] if len(kalshi_key) >= 4 else "****",
    }


@app.get("/api/settings")
def get_settings(admin: dict = Depends(require_admin)) -> dict:
    return _build_settings_response()


@app.patch("/api/settings")
def patch_settings(
    body: PatchSettingsBody, admin: dict = Depends(require_admin)
) -> dict:
    changes: list[str] = []
    if body.alpha_threshold is not None or body.lead_decay_halflife_hours is not None:
        cfg = _load_yaml()
        if "risk" not in cfg:
            cfg["risk"] = {}
        if body.alpha_threshold is not None:
            changes.append(
                f"risk.alpha_threshold {cfg['risk'].get('alpha_threshold')} "
                f"-> {body.alpha_threshold}"
            )
            cfg["risk"]["alpha_threshold"] = body.alpha_threshold
        if body.lead_decay_halflife_hours is not None:
            changes.append(
                f"risk.lead_decay_halflife_hours "
                f"{cfg['risk'].get('lead_decay_halflife_hours')} "
                f"-> {body.lead_decay_halflife_hours}"
            )
            cfg["risk"]["lead_decay_halflife_hours"] = body.lead_decay_halflife_hours
        _save_yaml(cfg)

    if body.discord_webhook_url is not None:
        changes.append("DISCORD_WEBHOOK_URL updated")
        _write_env_key("DISCORD_WEBHOOK_URL", body.discord_webhook_url)

    if changes:
        conn = _get_db()
        record_audit(conn, admin["username"], "settings.patch", detail="; ".join(changes))
        conn.close()

    return _build_settings_response()

# ---------------------------------------------------------------------------
# EXISTING ENDPOINTS — now protected with require_admin
# ---------------------------------------------------------------------------

@app.get("/api/summary")
def summary(admin: dict = Depends(require_admin)) -> dict[str, Any]:
    df = _load_trades()

    settled   = df[df["status"].isin(["WIN", "LOSS"])].copy()
    wins      = int((settled["status"] == "WIN").sum())
    losses    = int((settled["status"] == "LOSS").sum())
    n_settled = len(settled)

    win_rate  = (wins / n_settled) if n_settled > 0 else None
    net_pnl   = float(settled["realized_pnl"].fillna(0.0).sum()) if n_settled > 0 else 0.0

    all_alpha = pd.to_numeric(df["alpha"], errors="coerce").dropna()
    avg_alpha = float(all_alpha.mean()) if not all_alpha.empty else 0.0

    best_pnl  = float(settled["realized_pnl"].fillna(0.0).max()) if n_settled > 0 else 0.0
    worst_pnl = float(settled["realized_pnl"].fillna(0.0).min()) if n_settled > 0 else 0.0

    sharpe: float | None = None
    if n_settled >= 5 and "date" in settled.columns:
        daily = settled.groupby("date")["realized_pnl"].sum()
        std   = float(daily.std())
        if std > 0:
            sharpe = float(daily.mean() / std * math.sqrt(252))

    open_count = int((df["status"] == "OPEN").sum())
    return {
        "total_trades":    len(df),
        "open_count":      open_count,
        "open_positions":  open_count,
        "wins":            wins,
        "losses":          losses,
        "win_rate":        win_rate,
        "net_pnl":         net_pnl,
        "total_pnl":       net_pnl,
        "avg_alpha":       avg_alpha,
        "best_trade_pnl":  best_pnl,
        "worst_trade_pnl": worst_pnl,
        "sharpe":          sharpe,
    }


@app.get("/api/trades")
def trades(
    status: str = Query(default="ALL", description="OPEN|WIN|LOSS|VOID|ALL"),
    limit:  int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    format: str = Query(default="json", description="json|xlsx"),
    admin: dict = Depends(require_admin),
):
    valid_statuses = {"OPEN", "WIN", "LOSS", "VOID", "ALL"}
    if status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"status must be one of {sorted(valid_statuses)}",
        )
    df = _load_trades()
    if status != "ALL":
        df = df[df["status"] == status]
    df = df.sort_values("date", ascending=False, kind="stable")

    if format == "xlsx":
        # Full filtered set — a download is a report, not a page.
        return xlsx_response(df, f"trades_{status.lower()}", sheet_name="trades")

    total = len(df)
    page  = df.iloc[offset : offset + limit]
    return {
        "total":  total,
        "offset": offset,
        "limit":  limit,
        "trades": [_row_to_dict(row) for _, row in page.iterrows()],
    }


@app.get("/api/daily_log")
def daily_log(
    date_param: str | None = Query(default=None, alias="date"),
    format: str = Query(default="json", description="json|xlsx"),
    admin: dict = Depends(require_admin),
):
    if date_param is not None:
        try:
            date.fromisoformat(date_param)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid date: {date_param!r}")
        target = date_param
    else:
        target = str(date.today())

    df   = _load_daily_log()
    rows = df[df["date"].astype(str) == target]

    if format == "xlsx":
        return xlsx_response(rows, f"daily_log_{target}", sheet_name="daily_log")

    return {
        "date":  target,
        "count": len(rows),
        "rows":  [_row_to_dict(row) for _, row in rows.iterrows()],
    }


@app.get("/api/audit")
def audit(
    limit:  int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    username: str | None = Query(default=None),
    format: str = Query(default="json", description="json|xlsx"),
    admin: dict = Depends(require_admin),
):
    """Append-only audit trail (logins, config changes, user/trade edits)."""
    conn = _get_db()
    rows = fetch_audit(conn, limit=limit, offset=offset, username=username)
    conn.close()
    if format == "xlsx":
        return xlsx_response(pd.DataFrame(rows), "audit_log", sheet_name="audit")
    return {"count": len(rows), "rows": rows}


@app.get("/api/pnl_curve")
def pnl_curve(admin: dict = Depends(require_admin)) -> list[dict[str, Any]]:
    df      = _load_trades()
    settled = df[df["status"].isin(["WIN", "LOSS"])].copy()
    if settled.empty:
        return []
    settled["realized_pnl"] = settled["realized_pnl"].fillna(0.0)
    settled = settled.sort_values("date", ascending=True, kind="stable")
    daily   = (
        settled.groupby("date")
        .agg(daily_pnl=("realized_pnl", "sum"), win=("status", lambda s: (s == "WIN").any()))
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
def alpha_distribution(admin: dict = Depends(require_admin)) -> list[dict[str, Any]]:
    df     = _load_trades()
    alphas = pd.to_numeric(df["alpha"], errors="coerce").dropna()
    if alphas.empty:
        return []
    n_bins     = 20
    counts, _  = pd.cut(alphas, bins=n_bins, retbins=True)  # type: ignore[assignment]
    bin_counts = counts.value_counts(sort=False)
    result     = []
    for interval, count in bin_counts.items():
        lo  = interval.left   # type: ignore[union-attr]
        hi  = interval.right  # type: ignore[union-attr]
        mid = (lo + hi) / 2
        result.append({
            "bucket_label": f"[{lo:.3f}, {hi:.3f})",
            "count":        int(count),
            "is_positive":  bool(mid >= 0),
        })
    return result


_MONTH_NAMES = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]


@app.get("/api/bias_profile")
def bias_profile(admin: dict = Depends(require_admin)) -> dict[str, Any]:
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
            "month":               int(month_num),
            "month_name":          month_name,
            "mean_bias_f":         float(row.get("mean_bias_f", 0.0)),
            "variance_multiplier": float(row.get("variance_multiplier", 1.0)),
            "sample_count":        int(row.get("sample_count", 0)),
            "last_updated":        last_updated,
        })
    rows.sort(key=lambda r: r["month"])
    return {"source": "KMDW_monthly_profile.parquet", "rows": rows}


@app.get("/api/account")
def get_account(admin: dict = Depends(require_admin)) -> dict:
    try:
        from deep_isobar.market.kalshi_client import get_balance
        result = get_balance()
        if result is None:
            return {"balance_usd": None, "portfolio_value_usd": None, "total_usd": None,
                    "error": "Kalshi API unavailable"}
        return result
    except Exception as exc:
        return {"balance_usd": None, "portfolio_value_usd": None, "total_usd": None,
                "error": str(exc)}


class TradeOverride(BaseModel):
    status: str
    settled_temp: float | None = None


@app.patch("/api/trades/{contract_ticker}")
def override_trade(
    contract_ticker: str,
    body: TradeOverride,
    admin: dict = Depends(require_admin),
) -> dict[str, Any]:
    valid_statuses = {"WIN", "LOSS", "VOID"}
    if body.status not in valid_statuses:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(valid_statuses)}")

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
        df.loc[idx, "realized_pnl"] = round(
            _compute_pnl(direction, entry_price, body.status, position_size), 6
        )
    elif body.status == "VOID":
        df.loc[idx, "realized_pnl"] = 0.0

    df.to_csv(_PAPER_TRADES_CSV, index=False)
    conn = _get_db()
    record_audit(
        conn, admin["username"], "trades.patch",
        detail=f"{contract_ticker}: status -> {body.status}"
        + (f", settled_temp -> {body.settled_temp}" if body.settled_temp is not None else ""),
    )
    conn.close()
    return _row_to_dict(df.loc[idx])
