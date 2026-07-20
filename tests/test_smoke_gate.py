"""Smoke gate tests — METAR present-weather parsing + preflight check 7.

Motivated by the 2026-07-14..17 Canadian-wildfire slump: smoke attenuated
insolation and NE actual highs came in 5-10°F below all guidance.  The gate
stands a city down when recent METARs report FU (smoke) or HZ (haze) with
low visibility.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from deep_isobar.anomaly.metar_fetcher import parse_metar_fields
from deep_isobar.calibration.emos import EMOSParams
from deep_isobar.trading import preflight as pf
from deep_isobar.trading.preflight import SmokeReport, run_preflight, smoke_report

MODELS = ["GFS", "ECMWF", "ICON", "GEM", "NBM"]


def _params() -> EMOSParams:
    fitted = datetime.now(timezone.utc) - timedelta(hours=1)
    return EMOSParams(
        station_id="KTST", model_names=MODELS,
        a0=0.0, a=[0.2] * 5, c=1.0, d=0.5,
        fitted_at_utc=fitted.isoformat(timespec="seconds"),
    )


def _ok_kwargs(**overrides) -> dict:
    base = dict(
        city_name="Testville",
        effective_mean=90.0,
        effective_std=1.7,
        dist_source="EMOS",
        nbm_max_f=89.5,
        emos_params=_params(),
        n_contracts=10,
        market_is_live=True,
        hist_min_f=45.0,
        hist_max_f=98.0,
    )
    base.update(overrides)
    return base


def _clear_report() -> SmokeReport:
    return SmokeReport(n_obs=6, n_smoke=0, n_haze_low_vis=0, n_haze=0, min_vis_mi=10.0)


# ── METAR present-weather parsing ────────────────────────────────────────────


def test_parse_wx_smoke():
    fields = parse_metar_fields(
        "KNYC 161251Z 04008KT 4SM FU SCT250 31/18 A3002 RMK AO2"
    )
    assert fields["wx_codes"] == ["FU"]
    assert fields["visibility_mi"] == 4.0


def test_parse_wx_haze():
    fields = parse_metar_fields(
        "KBOS 151254Z 22010KT 5SM HZ FEW060 33/21 A2998 RMK AO2"
    )
    assert fields["wx_codes"] == ["HZ"]


def test_parse_wx_compound_and_intensity():
    fields = parse_metar_fields(
        "KDFW 141953Z 17012KT 3SM +TSRA BR BKN045CB 28/24 A2990 RMK AO2"
    )
    assert fields["wx_codes"] == ["TS", "RA", "BR"]


def test_parse_wx_clear_day_has_no_codes():
    fields = parse_metar_fields(
        "KMDW 101053Z 34007KT 10SM FEW018 OVC250 08/02 A2994 RMK AO2"
    )
    assert fields["wx_codes"] == []


def test_parse_wx_ignores_remarks_section():
    # FU in remarks (e.g. "FU ALQDS" commentary) must not count — only the body.
    fields = parse_metar_fields(
        "KSEA 161253Z 01004KT 10SM SKC 22/12 A3010 RMK AO2 FU ALQDS"
    )
    assert fields["wx_codes"] == []


def test_parse_wx_vicinity_prefix():
    fields = parse_metar_fields("KLAS 161256Z 10SM VCFU CLR 39/08 A2992")
    assert fields["wx_codes"] == ["FU"]


# ── SmokeReport aggregation (stubbed fetch, no network) ──────────────────────


def test_smoke_report_counts_and_min_vis(monkeypatch):
    metars = [
        "KNYC 161251Z 04008KT 4SM FU SCT250 31/18 A3002",
        "KNYC 161151Z 04008KT 5SM HZ SCT250 30/18 A3002",
        "KNYC 161051Z 04008KT 8SM HZ SCT250 29/18 A3002",
        "KNYC 160951Z 04008KT 10SM CLR 28/18 A3002",
    ]
    monkeypatch.setattr(
        "deep_isobar.anomaly.metar_fetcher.fetch_metars", lambda s, hours=6: metars
    )
    report = smoke_report("KNYC")
    assert report is not None
    assert report.n_obs == 4
    assert report.n_smoke == 1
    assert report.n_haze == 2
    assert report.n_haze_low_vis == 1   # 5SM HZ; the 8SM HZ is above threshold
    assert report.min_vis_mi == 4.0


def test_smoke_report_none_on_fetch_failure(monkeypatch):
    monkeypatch.setattr(
        "deep_isobar.anomaly.metar_fetcher.fetch_metars", lambda s, hours=6: []
    )
    assert smoke_report("KNYC") is None


def test_smoke_report_none_when_gate_disabled(monkeypatch):
    monkeypatch.setattr(pf, "_cfg", lambda key: False if key == "smoke_gate" else 6)
    assert smoke_report("KNYC") is None


# ── Preflight check 7 ────────────────────────────────────────────────────────


def test_preflight_blocks_on_smoke():
    smoke = SmokeReport(n_obs=6, n_smoke=2, n_haze_low_vis=0, n_haze=0, min_vis_mi=4.0)
    result = run_preflight(**_ok_kwargs(smoke=smoke))
    assert not result.ok
    assert any("smoke" in f for f in result.failures)


def test_preflight_blocks_on_low_vis_haze():
    smoke = SmokeReport(n_obs=6, n_smoke=0, n_haze_low_vis=3, n_haze=3, min_vis_mi=5.0)
    result = run_preflight(**_ok_kwargs(smoke=smoke))
    assert not result.ok
    assert any("smoke" in f for f in result.failures)


def test_preflight_warns_on_haze_with_good_vis():
    smoke = SmokeReport(n_obs=6, n_smoke=0, n_haze_low_vis=0, n_haze=2, min_vis_mi=8.0)
    result = run_preflight(**_ok_kwargs(smoke=smoke))
    assert result.ok
    assert any("haze" in w for w in result.warnings)


def test_preflight_warns_when_report_unavailable():
    result = run_preflight(**_ok_kwargs(smoke=None))
    assert result.ok
    assert any("smoke gate" in w for w in result.warnings)


def test_preflight_clear_day_passes_clean():
    result = run_preflight(**_ok_kwargs(smoke=_clear_report()))
    assert result.ok
    assert not any("smoke" in w or "haze" in w for w in result.warnings)
