# Deep Isobar Dashboard — Setup Guide

## Prerequisites

- Python 3.10+ with project virtualenv activated
- Node.js 18+

---

## 1. Seed the Admin User

The admin account is created automatically on first server startup if no admin exists in the database (`data/users.db`).

Add these variables to your `.env` file (copy from `.env.example`):

```
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password_here
JWT_SECRET=<64-char random hex string>
```

Generate a JWT secret:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

The admin account is created with `must_change_password=0`, so you can log in immediately with the credentials above.

---

## 2. Start the Backend

```bash
# From the project root with virtualenv active:
uvicorn deep_isobar.dashboard.api:app --reload --port 8765
```

On first run you will see:
```
[deep-isobar-api] Admin user seeded from .env  (username: admin)
```

---

## 3. Start the Frontend

```bash
cd dashboard_ui
npm install   # only needed once after pulling new deps
npm run dev
```

The dashboard is available at: http://localhost:5173

---

## 4. Create Investor Accounts

1. Log in at `/login` with your admin credentials
2. Navigate to **SETTINGS → USER_MANAGEMENT**
3. Click **+ CREATE_INVESTOR**
4. Fill in username, display name, email (optional), and a temporary password
5. Share the username and temporary password with the investor
6. The investor will be prompted to change their password on first login

---

## 5. Auth Details

| Property | Value |
|----------|-------|
| Token storage | `localStorage` key: `di_token` |
| Token type | JWT (HS256) |
| Token lifetime | 24 hours |
| Auto-refresh | Disabled — user must re-login after expiry |
| Roles | `admin`, `investor` |
| Admin can access | All pages including `/investor/*` |
| Investor can access | `/investor/*` only |

---

## 6. Route Summary

| Route | Access | Description |
|-------|--------|-------------|
| `/login` | Public | Login page |
| `/change-password` | Authenticated | Force-change password |
| `/` | Admin | Main dashboard (EXPOSURE/TERMINAL/WEATHER/BIAS tabs) |
| `/alerts` | Admin | Alert configuration |
| `/settings` | Admin | System config + user management |
| `/investor` | Investor + Admin | Portfolio overview |
| `/investor/performance` | Investor + Admin | Monthly/yearly P&L |
| `/investor/documents` | Investor + Admin | Document vault |
| `/investor/settings` | Investor + Admin | Account settings |

---

## 7. API Endpoints

All endpoints except `GET /api/health` require a JWT Bearer token.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | None | Watchdog health check |
| POST | `/api/auth/login` | None | Get JWT token |
| GET | `/api/auth/me` | Any | Current user info |
| POST | `/api/auth/change-password` | Any | Change password |
| GET | `/api/users` | Admin | List all users |
| POST | `/api/users` | Admin | Create investor |
| PATCH | `/api/users/{id}` | Admin | Update user |
| DELETE | `/api/users/{id}` | Admin | Delete user |
| GET | `/api/investor/summary` | Investor+ | Portfolio summary |
| GET | `/api/investor/performance` | Investor+ | Monthly/yearly stats |
| GET | `/api/alerts` | Any auth | List alerts |
| POST | `/api/alerts` | Admin | Create alert |
| PATCH | `/api/alerts/{id}` | Admin | Update alert |
| DELETE | `/api/alerts/{id}` | Admin | Delete alert |
| GET | `/api/settings` | Admin | Read system settings |
| PATCH | `/api/settings` | Admin | Update settings |
| GET | `/api/summary` | Admin | Trade summary stats |
| GET | `/api/trades` | Admin | Trade list |
| PATCH | `/api/trades/{ticker}` | Admin | Override trade status |
| GET | `/api/daily_log` | Admin | Daily model log |
| GET | `/api/pnl_curve` | Admin | Cumulative P&L data |
| GET | `/api/alpha_distribution` | Admin | Alpha histogram |
| GET | `/api/bias_profile` | Admin | KMDW bias profile |
| GET | `/api/account` | Admin | Kalshi balance |
