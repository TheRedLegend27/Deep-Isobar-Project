"""Golden-fixture tests: the contract pipeline against RECORDED Kalshi payloads.

Five bugs in one family (halved bracket math, never-firing settlement,
probability key collisions, T-ticker direction guessing, series-metadata
stub fallback) shared a root cause: contract semantics were *assumed*, not
verified against what the exchange actually returns.  These tests replay
real API responses — recorded by ``scripts/record_kalshi_fixtures.py`` —
through the production parse → probability → settlement code:

- every real open market must parse (no silent drops, all prefix families);
- parsed fields must be coherent and collision-free;
- a full bracket ladder's probabilities must sum to exactly 1;
- ``probability_for_contract`` and ``realized_yes_outcome`` must agree at
  every integer temperature (probability and settlement never diverge);
- our grading must reproduce the exchange's official ``result`` on settled
  contracts, including cap-boundary cases (B76.5 paid YES at exactly 77°F).

Fixtures are committed; re-record with the script when Kalshi's formats
change and these tests will tell you what broke.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import deep_isobar.market.kalshi_client as kc
from deep_isobar.models.probability_engine import probability_for_contract
from deep_isobar.research.settle_paper_trades import realized_yes_outcome

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "kalshi"
_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)

_MARKET_FIXTURES = sorted(FIXTURE_DIR.glob("markets_*.json"))
_ORDERBOOK_FIXTURES = sorted(FIXTURE_DIR.glob("orderbook_*.json"))
_SETTLED_FIXTURE = FIXTURE_DIR / "settled_markets.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_ids(paths: list[Path]) -> list[str]:
    return [p.stem for p in paths]


def test_fixtures_are_recorded():
    """The recorded fixtures must exist — see scripts/record_kalshi_fixtures.py."""
    assert len(_MARKET_FIXTURES) >= 3, "expected one markets fixture per prefix family"
    assert len(_ORDERBOOK_FIXTURES) >= 3, "expected one orderbook per strike type"
    assert _SETTLED_FIXTURE.exists()


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _MARKET_FIXTURES, ids=_fixture_ids(_MARKET_FIXTURES))
def test_every_real_open_market_parses(path):
    """No silent drops: every market the exchange serves must parse.

    A market parsing to None in production silently narrows the tradable
    universe (and once triggered the stub fallback for whole series).
    """
    markets = _load(path)["markets"]
    assert markets, f"{path.name} holds no markets — re-record"
    unparsed = [m["ticker"] for m in markets if kc._parse_contract(m, _NOW) is None]
    assert unparsed == [], f"real markets failed to parse: {unparsed}"


@pytest.mark.parametrize("path", _MARKET_FIXTURES, ids=_fixture_ids(_MARKET_FIXTURES))
def test_parsed_fields_are_coherent(path):
    markets = _load(path)["markets"]
    contracts = [kc._parse_contract(m, _NOW) for m in markets]
    cities = set()
    for c in contracts:
        assert c.strike_type in ("less", "greater", "between"), c.contract_id
        if c.strike_type == "less":
            assert c.cap_strike is not None, c.contract_id
        elif c.strike_type == "greater":
            assert c.floor_strike is not None, c.contract_id
        else:
            assert c.floor_strike is not None and c.cap_strike is not None
            assert c.floor_strike < c.cap_strike, c.contract_id
        cities.add(c.city)
    # One series maps to exactly one city.
    assert len(cities) == 1, f"{path.name} parsed to multiple cities: {cities}"
    # contract_id is the exchange ticker and unique.
    ids = [c.contract_id for c in contracts]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("path", _MARKET_FIXTURES, ids=_fixture_ids(_MARKET_FIXTURES))
def test_no_semantic_key_collisions(path):
    """(strike_type, floor, cap) is unique per target date.

    The dedup key that treated a T98 tail and the 98-99 bracket as
    duplicates lived exactly here.
    """
    markets = _load(path)["markets"]
    contracts = [kc._parse_contract(m, _NOW) for m in markets]
    keys = [(c.target_date, c.strike_type, c.floor_strike, c.cap_strike)
            for c in contracts]
    assert len(set(keys)) == len(keys)


# ---------------------------------------------------------------------------
# Probability — the ladder must be a probability distribution
# ---------------------------------------------------------------------------


def _ladder(path: Path):
    """Contracts of the fixture's first event (one target date), sorted."""
    markets = _load(path)["markets"]
    contracts = [kc._parse_contract(m, _NOW) for m in markets]
    first_date = min(c.target_date for c in contracts)
    ladder = [c for c in contracts if c.target_date == first_date]
    less = [c for c in ladder if c.strike_type == "less"]
    greater = [c for c in ladder if c.strike_type == "greater"]
    between = sorted(
        (c for c in ladder if c.strike_type == "between"),
        key=lambda c: c.floor_strike,
    )
    return less, between, greater


@pytest.mark.parametrize("path", _MARKET_FIXTURES, ids=_fixture_ids(_MARKET_FIXTURES))
def test_ladder_is_contiguous(path):
    """Tails + brackets partition the integer temperature line exactly."""
    less, between, greater = _ladder(path)
    assert len(less) == 1 and len(greater) == 1, "expected one tail on each side"
    assert between, "expected bracket contracts between the tails"
    assert less[0].cap_strike == between[0].floor_strike
    for prev, nxt in zip(between, between[1:]):
        assert nxt.floor_strike == prev.cap_strike + 1, (
            f"gap between brackets {prev.contract_id} and {nxt.contract_id}"
        )
    assert greater[0].floor_strike == between[-1].cap_strike


@pytest.mark.parametrize("path", _MARKET_FIXTURES, ids=_fixture_ids(_MARKET_FIXTURES))
@pytest.mark.parametrize("mu,sigma", [(70.0, 2.0), (85.0, 4.0), (100.0, 8.0)])
def test_ladder_probabilities_sum_to_one(path, mu, sigma):
    """Sum of YES probabilities across a full ladder == 1 for any forecast.

    The halved-bracket bug (exclusive cap) made this sum < 1 — every bracket
    leaked probability mass to nowhere.  This is the invariant that would
    have caught it on day one.
    """
    less, between, greater = _ladder(path)
    total = sum(
        probability_for_contract(c.strike_type, c.floor_strike, c.cap_strike, mu, sigma)
        for c in less + between + greater
    )
    assert total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Probability ↔ settlement consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _MARKET_FIXTURES, ids=_fixture_ids(_MARKET_FIXTURES))
def test_probability_and_settlement_agree_at_every_integer(path):
    """As sigma → 0, probability collapses onto the settlement outcome.

    Sweeps every integer temperature across the ladder range for every real
    contract: probability_for_contract must say ≈1 exactly where
    realized_yes_outcome pays and ≈0 where it doesn't.  Any drift between
    the pricing and grading conventions (the intraday cap-exclusive relapse)
    fails here.
    """
    less, between, greater = _ladder(path)
    lo = less[0].cap_strike - 3
    hi = greater[0].floor_strike + 3
    for c in less + between + greater:
        for temp in range(lo, hi + 1):
            p = probability_for_contract(
                c.strike_type, c.floor_strike, c.cap_strike, float(temp), 0.01
            )
            outcome = realized_yes_outcome(
                c.strike_type, c.floor_strike, c.cap_strike, float(temp)
            )
            assert p == pytest.approx(float(outcome), abs=1e-6), (
                f"{c.contract_id} at {temp}°F: probability {p:.4f} vs "
                f"settlement outcome {outcome}"
            )


# ---------------------------------------------------------------------------
# Settlement vs the exchange's official results
# ---------------------------------------------------------------------------


def test_settlement_matches_exchange_official_results():
    """Our grading reproduces Kalshi's ``result`` on real settled contracts.

    The fixture includes cap-boundary cases (B76.5 settled YES at exactly
    77°F; B88.5 settled YES at 89°F) — the exact family the exclusive-cap
    bugs got wrong twice.
    """
    entries = _load(_SETTLED_FIXTURE)["markets"]
    assert len(entries) >= 5, "settled fixture too thin — re-record"
    for entry in entries:
        m = entry["market"]
        ours = realized_yes_outcome(
            m["strike_type"], m.get("floor_strike"), m.get("cap_strike"),
            entry["settled_temp"],
        )
        exchange = 1 if m["result"] == "yes" else 0
        assert ours == exchange, (
            f"{m['ticker']}: our outcome {ours} vs exchange result "
            f"{m['result']!r} at settled_temp={entry['settled_temp']}"
        )


def test_settled_fixture_includes_cap_boundary_case():
    """Keep at least one settled_temp == cap_strike case in the fixture.

    If a re-record drops all boundary cases, the strongest assertion above
    silently weakens — fail loudly instead.
    """
    entries = _load(_SETTLED_FIXTURE)["markets"]
    assert any(
        e["market"]["strike_type"] == "between"
        and e["settled_temp"] == e["market"]["cap_strike"]
        for e in entries
    ), "re-record fixtures once a bracket settles exactly on its cap"


# ---------------------------------------------------------------------------
# Orderbook
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", _ORDERBOOK_FIXTURES, ids=_fixture_ids(_ORDERBOOK_FIXTURES)
)
def test_real_orderbooks_parse_to_sane_snapshots(path):
    ticker = path.stem.removeprefix("orderbook_")
    snap = kc._parse_orderbook_response(ticker, _load(path), _NOW)
    assert snap.contract_id == ticker
    for price in (snap.best_bid, snap.best_ask, snap.last_trade_price):
        if price is not None:
            assert 0.0 <= price <= 1.0, f"{ticker}: price {price} outside [0,1]"
    if snap.best_bid is not None and snap.best_ask is not None:
        assert snap.best_bid <= snap.best_ask, f"{ticker}: crossed book"
    for size in (snap.bid_size, snap.ask_size):
        if size is not None:
            assert size >= 0
