"""Security hardening: TOTP 2FA, JWT refresh/revocation, login rate limiting.

The internet-exposure prerequisites from docs/DASHBOARD_SPEC.md — each flow
is exercised end-to-end through the FastAPI TestClient.
"""

from __future__ import annotations

import os

import pyotp
import pytest

os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "test_admin_pw_123")

from fastapi.testclient import TestClient  # noqa: E402

import deep_isobar.dashboard.api as api  # noqa: E402
from deep_isobar.dashboard.api import app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Same isolation pattern as test_dashboard_rbac: force the seed creds and
    # point the module at a tmp users.db before _init_db.
    os.environ["ADMIN_USERNAME"] = "admin"
    os.environ["ADMIN_PASSWORD"] = "test_admin_pw_123"
    db = tmp_path_factory.mktemp("security") / "users.db"
    api._DB_PATH = db
    api._init_db()
    with TestClient(app) as c:
        yield c


def _login(client, username, password, **extra):
    return client.post("/api/auth/login",
                       json={"username": username, "password": password, **extra})


def _admin_token(client) -> str:
    r = _login(client, "admin", "test_admin_pw_123")
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _make_user(client, username, role="investor", password="pw12345") -> None:
    h = {"Authorization": f"Bearer {_admin_token(client)}"}
    r = client.post("/api/users", headers=h, json={
        "username": username, "display_name": username,
        "temp_password": password, "role": role,
    })
    assert r.status_code == 201, r.text


# ---------------------------------------------------------------------------
# JWT refresh + revocation
# ---------------------------------------------------------------------------

def test_login_returns_refresh_token_and_expiry(client):
    r = _login(client, "admin", "test_admin_pw_123")
    body = r.json()
    assert body["refresh_token"]
    assert body["expires_in"] == api._ACCESS_EXPIRE_MIN * 60
    # Admin has no TOTP enrolled yet — mandatory-role flag must be raised.
    assert body["totp_enrollment_required"] is True


def test_refresh_rotates_tokens(client):
    tokens = _login(client, "admin", "test_admin_pw_123").json()
    r = client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == 200, r.text
    fresh = r.json()
    assert fresh["access_token"] and fresh["refresh_token"]
    ok = client.get("/api/auth/me",
                    headers={"Authorization": f"Bearer {fresh['access_token']}"})
    assert ok.status_code == 200


def test_refresh_token_rejected_as_access_token(client):
    tokens = _login(client, "admin", "test_admin_pw_123").json()
    r = client.get("/api/auth/me",
                   headers={"Authorization": f"Bearer {tokens['refresh_token']}"})
    assert r.status_code == 401


def test_access_token_rejected_by_refresh_endpoint(client):
    tokens = _login(client, "admin", "test_admin_pw_123").json()
    r = client.post("/api/auth/refresh", json={"refresh_token": tokens["access_token"]})
    assert r.status_code == 401


def test_password_change_revokes_old_tokens(client):
    _make_user(client, "rotate1", password="oldpw123")
    old = _login(client, "rotate1", "oldpw123").json()
    h_old = {"Authorization": f"Bearer {old['access_token']}"}

    r = client.post("/api/auth/change-password", headers=h_old,
                    json={"current_password": "oldpw123", "new_password": "newpw12345"})
    assert r.status_code == 200, r.text
    fresh = r.json()

    # Old pair is dead — both directly and via refresh.
    assert client.get("/api/auth/me", headers=h_old).status_code == 401
    assert client.post("/api/auth/refresh",
                       json={"refresh_token": old["refresh_token"]}).status_code == 401
    # The pair returned by change-password works.
    assert client.get("/api/auth/me", headers={
        "Authorization": f"Bearer {fresh['access_token']}"}).status_code == 200


def test_admin_password_reset_revokes_user_tokens(client):
    _make_user(client, "rotate2", password="oldpw123")
    victim = _login(client, "rotate2", "oldpw123").json()
    h_admin = {"Authorization": f"Bearer {_admin_token(client)}"}

    users = client.get("/api/users", headers=h_admin).json()
    uid = next(u["id"] for u in users if u["username"] == "rotate2")
    r = client.patch(f"/api/users/{uid}", headers=h_admin,
                     json={"temp_password": "resetpw123"})
    assert r.status_code == 200, r.text

    assert client.get("/api/auth/me", headers={
        "Authorization": f"Bearer {victim['access_token']}"}).status_code == 401


# ---------------------------------------------------------------------------
# TOTP 2FA
# ---------------------------------------------------------------------------

def test_totp_full_lifecycle(client):
    _make_user(client, "totper", role="cio", password="pw12345")
    tok = _login(client, "totper", "pw12345").json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}

    # Setup returns a secret + provisioning URI; not yet enforced.
    r = client.post("/api/auth/totp/setup", headers=h)
    assert r.status_code == 200, r.text
    secret = r.json()["secret"]
    assert "otpauth://" in r.json()["otpauth_uri"]
    assert _login(client, "totper", "pw12345").status_code == 200

    # Enable requires a valid code.
    bad = client.post("/api/auth/totp/enable", headers=h, json={"code": "000000"})
    assert bad.status_code == 400
    good = client.post("/api/auth/totp/enable", headers=h,
                       json={"code": pyotp.TOTP(secret).now()})
    assert good.status_code == 200, good.text

    # Login without a code → 401 with the totp_required flag.
    r = _login(client, "totper", "pw12345")
    assert r.status_code == 401
    assert r.json()["detail"]["totp_required"] is True

    # Wrong code → 401; right code → full login with totp_enabled.
    assert _login(client, "totper", "pw12345", totp_code="000000").status_code == 401
    r = _login(client, "totper", "pw12345", totp_code=pyotp.TOTP(secret).now())
    assert r.status_code == 200, r.text
    assert r.json()["totp_enabled"] is True
    assert r.json()["totp_enrollment_required"] is False

    # Disable needs password AND code; then plain login works again.
    tok2 = r.json()["access_token"]
    h2 = {"Authorization": f"Bearer {tok2}"}
    assert client.post("/api/auth/totp/disable", headers=h2, json={
        "code": pyotp.TOTP(secret).now(), "password": "wrong"}).status_code == 400
    assert client.post("/api/auth/totp/disable", headers=h2, json={
        "code": pyotp.TOTP(secret).now(), "password": "pw12345"}).status_code == 200
    assert _login(client, "totper", "pw12345").status_code == 200


def test_totp_secret_never_leaks(client):
    h = {"Authorization": f"Bearer {_admin_token(client)}"}
    for u in client.get("/api/users", headers=h).json():
        assert "totp_secret" not in u and "hashed_password" not in u
    me = client.get("/api/auth/me", headers=h).json()
    assert "totp_secret" not in me


# ---------------------------------------------------------------------------
# Login rate limiting
# ---------------------------------------------------------------------------

def test_per_username_lockout(client):
    _make_user(client, "bruteforce", password="pw12345")
    for _ in range(api._LOGIN_MAX_FAILS_USER):
        assert _login(client, "bruteforce", "wrongpw").status_code == 401
    # Locked out now — even with the correct password.
    r = _login(client, "bruteforce", "pw12345")
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    # Other accounts are unaffected (per-IP limit is higher).
    assert _login(client, "admin", "test_admin_pw_123").status_code == 200
