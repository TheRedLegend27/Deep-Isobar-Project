"""RBAC: the role→scope matrix must gate every endpoint correctly."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "test_admin_pw_123")

from fastapi.testclient import TestClient  # noqa: E402

import deep_isobar.dashboard.api as api  # noqa: E402
from deep_isobar.dashboard.api import ROLE_SCOPES, app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Isolate from the real data/users.db — seed a fresh admin we control.
    # Force the seed creds (not setdefault): another test module may have
    # already set ADMIN_* in os.environ, and import order is not guaranteed.
    os.environ["ADMIN_USERNAME"] = "admin"
    os.environ["ADMIN_PASSWORD"] = "test_admin_pw_123"
    db = tmp_path_factory.mktemp("rbac") / "users.db"
    api._DB_PATH = db
    api._init_db()
    # The lifespan re-runs _init_db on the same (now-tmp) path — idempotent.
    with TestClient(app) as c:
        yield c


def _admin_token(client) -> str:
    r = client.post("/api/auth/login",
                    json={"username": "admin", "password": "test_admin_pw_123"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _make_user(client, admin_tok, username, role) -> str:
    h = {"Authorization": f"Bearer {admin_tok}"}
    client.post("/api/users", headers=h, json={
        "username": username, "display_name": username,
        "temp_password": "pw12345", "role": role,
    })
    # New users must change password on first login, but the token still works.
    r = client.post("/api/auth/login", json={"username": username, "password": "pw12345"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def test_me_exposes_scopes(client):
    tok = _admin_token(client)
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tok}"}).json()
    assert set(me["scopes"]) == ROLE_SCOPES["admin"]


def test_admin_cannot_be_created_via_api(client):
    admin_tok = _admin_token(client)
    h = {"Authorization": f"Bearer {admin_tok}"}
    r = client.post("/api/users", headers=h, json={
        "username": "sneaky", "display_name": "x",
        "temp_password": "pw12345", "role": "admin",
    })
    assert r.status_code == 422


def test_investor_reaches_reports_not_trading(client):
    admin_tok = _admin_token(client)
    tok = _make_user(client, admin_tok, "inv1", "investor")
    h = {"Authorization": f"Bearer {tok}"}
    assert client.get("/api/investor/summary", headers=h).status_code == 200
    assert client.get("/api/positions", headers=h).status_code == 403
    assert client.get("/api/scorecard", headers=h).status_code == 403


def test_cio_reaches_calibration_not_admin(client):
    admin_tok = _admin_token(client)
    tok = _make_user(client, admin_tok, "cio1", "cio")
    h = {"Authorization": f"Bearer {tok}"}
    assert client.get("/api/scorecard", headers=h).status_code == 200
    assert client.get("/api/positions", headers=h).status_code == 200
    assert client.get("/api/users", headers=h).status_code == 403     # admin only
    assert client.get("/api/settings", headers=h).status_code == 403


def test_it_reaches_nothing_but_ops(client):
    admin_tok = _admin_token(client)
    tok = _make_user(client, admin_tok, "it1", "it")
    h = {"Authorization": f"Bearer {tok}"}
    # No ops data endpoint ships yet, but the role must be blocked everywhere else.
    assert client.get("/api/positions", headers=h).status_code == 403
    assert client.get("/api/scorecard", headers=h).status_code == 403
    assert client.get("/api/investor/summary", headers=h).status_code == 403


def test_unauthenticated_is_401(client):
    assert client.get("/api/positions").status_code == 401
    assert client.get("/api/scorecard").status_code == 401
