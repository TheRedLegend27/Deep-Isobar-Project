"""Historical GFS forecast ingestion from NOAA's AWS Open Data archive.

Pulls archived GFS 0.25° forecast runs for a given city and calendar month,
extracting just the 2 m temperature at the city's nearest grid point via
HTTP byte-range requests against the GRIB2 index (`.idx`) files.  This
avoids downloading the full 700 MB–2 GB GRIB2 files by fetching only the
specific variable record (~1–3 MB each).

Downloads are cached on disk so repeated runs never re-fetch the same data.

Archive source
--------------
AWS Open Data Registry (public, no authentication required):

    https://noaa-gfs-bdp-pds.s3.amazonaws.com/

NOAA inserted an ``atmos/`` subdirectory into the GFS archive path during a
gradual migration.  The exact cutover varies by date and run; both layouts
coexist for an indeterminate period.  ``_resolve_gfs_idx_url`` handles this
transparently by trying ``atmos/`` first, then the bare path, returning the
URL that returns HTTP 200.  If both return 404 the run is logged as an
archive gap and skipped gracefully.

  With atmos/:  gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF[.idx]
  Without:      gfs.YYYYMMDD/HH/gfs.tHHz.pgrb2.0p25.fFFF[.idx]

Coverage: confirmed available from roughly 2021 onward; some earlier dates
exist but with more gaps.  August 2023 is confirmed complete.

Lead-time schedule (18z UTC ≈ 1 pm CDT, afternoon-high proxy)
--------------------------------------------------------------
00z run: f042, f066, f090, f114, f138  →  target days +1 … +5
12z run: f030, f054, f078, f102, f126  →  target days +1 … +5

Required dependencies (not in base requirements.txt)
-----------------------------------------------------
::

    pip install eccodes cfgrib xarray

``eccodes`` ships pre-built wheels for Windows/macOS/Linux as of 2023; no
conda or system installation is needed.  ``cfgrib`` wraps eccodes;
``xarray`` provides the ``.sel(method='nearest')`` point-extraction API.

Public interface::

    fetch_gfs_forecasts(city, year, month, ...) -> pd.DataFrame
    save_forecast_parquet(df, output_path)      -> Path

Output conforms to the ``forecast_temperature_point`` schema
(``docs/DATA_SCHEMA.md`` §7).

Usage::

    python -m deep_isobar.data.historical_forecast_ingest \\
        --city Chicago --year 2023 --month 8 \\
        --out data/historical/forecasts/gfs_chicago_2023.parquet

or::

    python src/deep_isobar/data/historical_forecast_ingest.py
"""

from __future__ import annotations

import calendar
import logging
import random
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from deep_isobar.data.city_universe import get_city_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# KMDW (Chicago Midway) coordinates — Kalshi KXHIGHCHI settles on Midway.
# GFS uses 0–360 longitude convention: lon_360 = 360 − west_longitude.
_KMDW_LAT = 41.7868
_KMDW_LON_360 = 272.2478  # 360 - 87.7522 = 272.2478°E

# Station coordinates for GFS grid-point extraction.
# Longitude stored in 0-360 convention (= 360 + west_longitude).
# Add new stations here when expanding to additional cities.
_STATION_COORDS: dict[str, tuple[float, float]] = {
    "KMDW": (_KMDW_LAT, _KMDW_LON_360),           # Chicago Midway (Kalshi settlement)
    "KORD": (41.9803, 272.0911),                   # Chicago O'Hare (kept for reference)
    "KJFK": (40.6413, 286.2219),                   # New York JFK
    "KPHX": (33.4373, 247.9922),                   # Phoenix Sky Harbor
    "KDEN": (39.8561, 255.3263),                   # Denver International
    "KDFW": (32.8998, 262.9597),                   # Dallas/Fort Worth
}

# Forecast hours that land at 18z UTC (≈ 1 pm CDT) for each lead day
# Structure: cycle → {lead_day: fhour}
_LEAD_FHOUR: dict[str, dict[int, int]] = {
    "00": {1: 42,  2: 66,  3: 90,  4: 114, 5: 138},
    "12": {1: 30,  2: 54,  3: 78,  4: 102, 5: 126},
}

# Output columns in DATA_SCHEMA.md §7 order
_OUTPUT_COLUMNS = [
    "city",
    "station_id",
    "model_name",
    "run_time_utc",
    "target_date",
    "metric",
    "forecast_value_f",
    "lead_hours",
    "source_name",
    "ingestion_time_utc",
]

_RETRY_STATUSES = {429, 500, 502, 503, 504}

# Serialize cfgrib opens across threads.  Chicago and Dallas share the same
# cached GRIB2 snippets (city-agnostic grid files).  When two threads call
# cfgrib.open_dataset() on the same file simultaneously, one regenerates the
# stale .*.idx sidecar while the other's eccodes context is mid-read, causing:
#   ECCODES ERROR: grib_handle_create: Cannot create handle, no definitions found
# The lock ensures only one thread opens/parses a GRIB file at a time; the
# scalar extraction (arr.flat[0]) and unit conversion are done outside the lock.
_GRIB_OPEN_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------


def _check_cfgrib() -> None:
    """Raise ``ImportError`` with install instructions if cfgrib/xarray are absent.

    Raises:
        ImportError: If either ``cfgrib`` or ``xarray`` cannot be imported,
            with a message containing the installation command.
    """
    missing = []
    try:
        import cfgrib  # noqa: F401
    except ImportError:
        missing.append("cfgrib")
    try:
        import xarray  # noqa: F401
    except ImportError:
        missing.append("xarray")
    try:
        import eccodes  # noqa: F401
    except ImportError:
        if "cfgrib" not in missing:
            missing.append("eccodes")

    if missing:
        raise ImportError(
            f"Missing required package(s): {', '.join(missing)}. "
            "Install with:\n\n    pip install eccodes cfgrib xarray\n"
        )


# ---------------------------------------------------------------------------
# URL and path helpers
# ---------------------------------------------------------------------------


def _resolve_gfs_idx_url(
    base_url: str,
    date_str: str,
    cycle: str,
    fhour: str,
) -> str | None:
    """Probe S3 for the correct ``.idx`` URL, trying ``atmos/`` then bare path.

    NOAA migrated the GFS archive layout gradually; the ``atmos/`` subdirectory
    was inserted between the cycle directory and the filename, but the exact
    cutover varies.  This function tries both layouts and returns whichever
    returns HTTP 200.

    Args:
        base_url: S3 bucket root (e.g. ``_AWS_BASE``).
        date_str: Run date formatted as ``"YYYYMMDD"``.
        cycle: Run cycle, ``"00"`` or ``"12"``.
        fhour: Zero-padded forecast hour string, e.g. ``"042"``.

    Returns:
        The first ``.idx`` URL that returns HTTP 200, or ``None`` if both
        return 404 (archive gap — caller should skip and warn).

    Example::

        idx_url = _resolve_gfs_idx_url(_AWS_BASE, "20230801", "00", "042")
        # → ".../gfs.20230801/00/atmos/gfs.t00z.pgrb2.0p25.f042.idx"
        grib2_url = idx_url[:-4]  # strip ".idx"
    """
    fname = f"gfs.t{cycle}z.pgrb2.0p25.f{fhour}.idx"
    for subdir in ("atmos/", ""):
        url = f"{base_url}/gfs.{date_str}/{cycle}/{subdir}{fname}"
        try:
            r = requests.head(url, timeout=10)
        except requests.RequestException:
            raise  # network error — don't silently try the other subdir
        if r.status_code == 200:
            return url
        # 404 → try next layout; any other status is also non-fatal for resolution
    return None


def _cache_path(cache_dir: Path, run_date: date, cycle: str, fhour: int) -> Path:
    """Return the canonical on-disk path for a cached GRIB2 snippet.

    Layout: ``cache_dir/gfs.YYYYMMDD/CC/tmp2m_fFFF.grib2``

    Args:
        cache_dir: Root cache directory.
        run_date: Model run date.
        cycle: ``"00"`` or ``"12"``.
        fhour: Forecast hour.

    Returns:
        Resolved :class:`~pathlib.Path` for the cached file.
    """
    return (
        cache_dir
        / f"gfs.{run_date.strftime('%Y%m%d')}"
        / cycle
        / f"tmp2m_f{fhour:03d}.grib2"
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get_with_backoff(
    url: str,
    headers: dict[str, str] | None = None,
    max_retries: int = 5,
    base_delay_s: float = 1.0,
    max_delay_s: float = 60.0,
) -> requests.Response:
    """GET *url* with exponential backoff and jitter.

    Retries on HTTP status codes 429, 500, 502, 503, 504 and on
    :class:`requests.ConnectionError` / :class:`requests.Timeout`.

    Delay formula::

        delay = min(base_delay_s * 2^attempt + random.uniform(0, 1), max_delay_s)

    Args:
        url: Target URL.
        headers: Optional extra headers (e.g. ``Range``).
        max_retries: Total attempts before re-raising.
        base_delay_s: Base delay in seconds.
        max_delay_s: Maximum delay cap in seconds.

    Returns:
        The successful :class:`requests.Response`.

    Raises:
        requests.HTTPError: Propagated after ``max_retries`` exhausted.
        RuntimeError: If all retries fail for non-HTTP reasons.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers or {}, timeout=30)
            if resp.status_code in _RETRY_STATUSES:
                raise requests.HTTPError(
                    f"HTTP {resp.status_code}", response=resp
                )
            resp.raise_for_status()
            return resp

        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == max_retries - 1:
                raise
            delay = min(
                base_delay_s * (2 ** attempt) + random.uniform(0, 1),
                max_delay_s,
            )
            logger.warning(
                "GET attempt %d/%d failed for %s (%s) — retrying in %.1fs",
                attempt + 1, max_retries, url, exc, delay,
            )
            time.sleep(delay)

    raise RuntimeError("Unexpected exit from retry loop")  # pragma: no cover


# ---------------------------------------------------------------------------
# IDX parsing
# ---------------------------------------------------------------------------


def _parse_tmp2m_byte_range(idx_text: str) -> tuple[int, int | None]:
    """Parse a GFS ``.idx`` file and return the byte range for ``TMP:2 m above ground``.

    Each line of the idx file has the format::

        RECNUM:BYTE_OFFSET:d=YYYYMMDDHH:VARIABLE:LEVEL:FCST_TYPE[:extra]

    This function finds the line where ``VARIABLE == "TMP"`` and
    ``LEVEL == "2 m above ground"``, then computes the byte span from
    that record's start to the byte before the next record begins (or
    ``None`` if this is the last record in the file).

    Args:
        idx_text: Full text content of the ``.idx`` file.

    Returns:
        ``(start_byte, end_byte)`` where ``end_byte`` is ``None`` for the
        last record (meaning: read to end of file).

    Raises:
        ValueError: If no matching record is found in the idx.

    Example::

        start, end = _parse_tmp2m_byte_range(idx_content)
        headers = {"Range": f"bytes={start}-{end}" if end else f"bytes={start}-"}
    """
    lines = [ln for ln in idx_text.strip().splitlines() if ln.strip()]

    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 6:
            continue
        variable = parts[3].strip()
        level = parts[4].strip()
        if variable == "TMP" and level == "2 m above ground":
            start_byte = int(parts[1])
            if i + 1 < len(lines):
                next_parts = lines[i + 1].split(":")
                end_byte: int | None = int(next_parts[1]) - 1
            else:
                end_byte = None
            return start_byte, end_byte

    raise ValueError("TMP:2 m above ground not found in idx")


# ---------------------------------------------------------------------------
# Download + cache
# ---------------------------------------------------------------------------


def _download_snippet(grib2_url: str, idx_url: str, dest: Path) -> None:
    """Download the TMP@2m GRIB2 record via idx byte-range into *dest*.

    If *dest* already exists, this is a no-op (cache hit).

    Args:
        grib2_url: Full HTTPS URL to the GRIB2 file on AWS.
        idx_url: Full HTTPS URL to the corresponding ``.idx`` file.
        dest: Target path for the cached GRIB2 snippet.

    Raises:
        requests.HTTPError: If the idx or GRIB2 fetch fails (caller decides
            whether to skip or abort).
        ValueError: If ``TMP:2 m above ground`` is absent from the idx.
    """
    if dest.exists():
        logger.debug("Cache hit: %s", dest)
        return

    logger.info("Fetching idx: %s", idx_url)
    idx_resp = _get_with_backoff(idx_url)
    idx_text = idx_resp.text

    start_byte, end_byte = _parse_tmp2m_byte_range(idx_text)
    range_header = (
        f"bytes={start_byte}-{end_byte}"
        if end_byte is not None
        else f"bytes={start_byte}-"
    )

    logger.info(
        "Fetching GRIB2 snippet: %s  Range: %s", grib2_url, range_header
    )
    grib_resp = _get_with_backoff(grib2_url, headers={"Range": range_header})

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(grib_resp.content)
    logger.debug(
        "Cached %d bytes to %s", len(grib_resp.content), dest
    )


# ---------------------------------------------------------------------------
# GRIB extraction
# ---------------------------------------------------------------------------


def _extract_fahrenheit(grib_path: Path, lat: float, lon_360: float) -> float:
    """Parse a cached GRIB2 snippet and return the temperature in °F at the nearest point.

    The GFS stores 2 m temperature in Kelvin.  This function:

    1. Opens *grib_path* with ``cfgrib``, filtering for
       ``typeOfLevel='heightAboveGround'``, ``level=2`` (2 m AGL).
    2. Selects the nearest grid point to ``(lat, lon_360)`` using
       ``xarray``'s ``sel(method='nearest')``.
    3. Converts Kelvin → Fahrenheit: ``°F = (K - 273.15) × 9/5 + 32``.

    Args:
        grib_path: Path to the cached GRIB2 snippet file.
        lat: Target latitude in decimal degrees (e.g. ``41.98``).
        lon_360: Target longitude in 0–360 convention
            (e.g. ``272.09`` for KORD).

    Returns:
        Temperature in °F at the nearest GFS grid point.

    Raises:
        ImportError: If cfgrib or xarray is not installed.
        Exception: Any cfgrib/xarray error reading the GRIB2 file.
    """
    import cfgrib  # noqa: PLC0415 — deferred, behind _check_cfgrib guard

    with _GRIB_OPEN_LOCK:
        # Purge stale cfgrib index sidecars before opening.  The sidecar
        # filename embeds a version hash (e.g. .da267.idx) that changes when
        # cfgrib is updated; leaving stale sidecars causes cfgrib to warn and
        # regenerate them.  Deleting them here is safe: cfgrib will recreate the
        # index on the next open.  Crucially, this must happen inside the lock so
        # that two threads never simultaneously try to regenerate the same index,
        # which causes eccodes to lose its handle context and raise:
        #   "Cannot create handle, no definitions found"
        for stale_idx in grib_path.parent.glob(f"{grib_path.name}.*.idx"):
            stale_idx.unlink(missing_ok=True)

        ds = cfgrib.open_dataset(
            str(grib_path),
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
            errors="raise",
        )

        # cfgrib names 2m temperature "t2m" (CF convention for height-above-ground)
        var = "t2m" if "t2m" in ds else "t"
        # .sel() with scalar lat/lon should return a 0-d DataArray, but cfgrib may
        # leave a residual `step` or `valid_time` dimension when the GRIB2 byte-range
        # snippet contains multiple records.  Evaluating such a multi-element numpy
        # array in a boolean or float() context raises:
        #   "The truth value of an array with more than one element is ambiguous"
        # or the equivalent "only size-1 arrays can be converted to Python scalars".
        # .squeeze() collapses all size-1 dimensions first; .flat[0] then extracts
        # a Python scalar safely regardless of the remaining array shape.
        arr = ds[var].sel(latitude=lat, longitude=lon_360, method="nearest").squeeze().values

    # Pure Python math — no need to hold the lock past the array extraction.
    t_k = float(arr.flat[0])
    t_f = (t_k - 273.15) * 9.0 / 5.0 + 32.0
    return round(t_f, 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_gfs_forecasts(
    city: str,
    year: int,
    month: int,
    cycles: list[str] | None = None,
    lead_days: list[int] | None = None,
    cache_dir: str | None = None,
    config_dir: str | None = None,
) -> pd.DataFrame:
    """Fetch archived GFS 0.25° forecasts for one city and calendar month.

    Iterates over every run date in ``year``/``month``, for each requested
    cycle (``"00"`` or ``"12"``), for each ``lead_day`` (1–5).  For each
    combination the function:

    1. Resolves the correct AWS path (tries ``atmos/`` then bare) via HEAD.
    2. Fetches the idx to find the byte offset of ``TMP:2 m above ground``.
    3. Downloads only that GRIB2 record via an HTTP ``Range`` request.
    4. Caches the snippet locally (subsequent runs are instant).
    5. Extracts the value at the nearest grid point to the city's station.
    6. Converts Kelvin → °F and appends a ``forecast_temperature_point`` row.

    The 18z UTC valid time is used as the "afternoon high" proxy for each
    target date (≈ 1 pm CDT for Chicago).

    Lead-hour mapping::

        cycle  lead_day → fhour (valid at 18z UTC)
        00z    1 → 42,  2 → 66,  3 → 90,  4 → 114, 5 → 138
        12z    1 → 30,  2 → 54,  3 → 78,  4 → 102, 5 → 126

    Runs that return a 404 from AWS (NOAA occasionally skips a cycle)
    are logged as warnings and skipped gracefully.  Corrupt cached files
    are deleted and re-downloaded once before being skipped.

    Args:
        city: Exact city name as in ``config/cities.yaml``
            (e.g. ``"Chicago"``).  Case-sensitive.
        year: Four-digit year (e.g. ``2023``).
        month: Calendar month 1–12 (e.g. ``8`` for August).
        cycles: GFS run cycles to include.  Defaults to ``["00", "12"]``.
        lead_days: Lead days to extract, each in 1–5.
            Defaults to ``[1, 2, 3, 4, 5]``.
        cache_dir: Directory for cached GRIB2 snippets.  Defaults to
            ``data/historical/forecasts/.grib_cache`` relative to the
            current working directory.
        config_dir: Optional override for the config directory containing
            ``cities.yaml``.

    Returns:
        DataFrame with columns conforming to
        ``forecast_temperature_point`` (DATA_SCHEMA.md §7):

        - ``city``, ``station_id``, ``model_name``
        - ``run_time_utc`` (datetime, UTC)
        - ``target_date`` (``datetime.date``)
        - ``metric`` = ``"high_temp_f"``
        - ``forecast_value_f`` (float, °F)
        - ``lead_hours`` (int)
        - ``source_name`` = ``"NOAA"``
        - ``ingestion_time_utc`` (datetime, UTC)

    Raises:
        ImportError: If ``cfgrib`` or ``xarray`` is not installed.
        KeyError: If *city* is not in ``cities.yaml``.
        RuntimeError: If every run in the month was skipped (e.g. no
            network connectivity).

    Example::

        from deep_isobar.data.historical_forecast_ingest import fetch_gfs_forecasts
        df = fetch_gfs_forecasts("Chicago", 2023, 8)
        print(df.shape)   # (~310, 10)
    """
    _check_cfgrib()

    cycles = cycles if cycles is not None else ["00", "12"]
    lead_days = lead_days if lead_days is not None else [1, 2, 3, 4, 5]

    profile = get_city_profile(city, config_dir=config_dir)
    station_id = profile.station_id
    city_lat, city_lon_360 = _STATION_COORDS.get(station_id, (_KMDW_LAT, _KMDW_LON_360))

    cache_root = (
        Path(cache_dir) if cache_dir
        else Path("data/historical/forecasts/.grib_cache")
    )

    ingestion_ts = datetime.now(timezone.utc)
    _, n_days = calendar.monthrange(year, month)

    logger.info(
        "=== GFS archive fetch: city=%s station=%s  %04d-%02d  "
        "cycles=%s  lead_days=%s ===",
        city, station_id, year, month, cycles, lead_days,
    )

    records: list[dict[str, Any]] = []
    skipped = 0
    total = n_days * len(cycles) * len(lead_days)

    for day in range(1, n_days + 1):
        run_date = date(year, month, day)
        date_str = run_date.strftime("%Y%m%d")

        for cycle in cycles:
            run_time_utc = datetime(
                year, month, day,
                int(cycle), 0, 0,
                tzinfo=timezone.utc,
            )

            for lead_day in lead_days:
                fhour = _LEAD_FHOUR[cycle][lead_day]
                fhour_str = f"{fhour:03d}"
                target_date = run_date + timedelta(days=lead_day)
                dest = _cache_path(cache_root, run_date, cycle, fhour)

                # ── Resolve URL and download if not cached ──────────────────
                grib2_url: str | None = None
                idx_url: str | None = None

                if not dest.exists():
                    try:
                        idx_url = _resolve_gfs_idx_url(
                            _AWS_BASE, date_str, cycle, fhour_str
                        )
                    except requests.RequestException as exc:
                        logger.warning(
                            "Network error resolving %s/%s f%s: %s — skipping",
                            run_date, cycle, fhour_str, exc,
                        )
                        skipped += 1
                        continue
                    if idx_url is None:
                        logger.warning(
                            "HTTP 404 for %s/%s f%s — archive gap, skipping",
                            run_date, cycle, fhour_str,
                        )
                        skipped += 1
                        continue
                    grib2_url = idx_url[:-4]  # strip ".idx"

                    try:
                        _download_snippet(grib2_url, idx_url, dest)
                    except requests.HTTPError as exc:
                        status = (
                            exc.response.status_code
                            if exc.response is not None
                            else "?"
                        )
                        logger.warning(
                            "HTTP %s for %s/%s f%s — skipping",
                            status, run_date, cycle, fhour_str,
                        )
                        skipped += 1
                        continue
                    except ValueError as exc:
                        logger.warning(
                            "idx parse error for %s/%s f%s: %s — skipping",
                            run_date, cycle, fhour_str, exc,
                        )
                        skipped += 1
                        continue

                # ── Extract temperature ─────────────────────────────────────
                try:
                    t_f = _extract_fahrenheit(dest, city_lat, city_lon_360)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "cfgrib error for %s  (%s) — deleting cache and retrying",
                        dest.name, exc,
                    )
                    dest.unlink(missing_ok=True)
                    # Re-resolve URL if we came from a cache hit (grib2_url is None).
                    if grib2_url is None:
                        try:
                            idx_url = _resolve_gfs_idx_url(
                                _AWS_BASE, date_str, cycle, fhour_str
                            )
                        except requests.RequestException:
                            idx_url = None
                        if idx_url is None:
                            logger.warning(
                                "Cannot re-resolve URL for retry of %s — skipping",
                                dest.name,
                            )
                            skipped += 1
                            continue
                        grib2_url = idx_url[:-4]
                    try:
                        _download_snippet(grib2_url, idx_url, dest)
                        t_f = _extract_fahrenheit(dest, city_lat, city_lon_360)
                    except Exception as retry_exc:  # noqa: BLE001
                        logger.warning(
                            "Retry also failed for %s (%s) — skipping",
                            dest.name, retry_exc,
                        )
                        skipped += 1
                        continue

                records.append(
                    {
                        "city": city,
                        "station_id": station_id,
                        "model_name": "GFS",
                        "run_time_utc": run_time_utc,
                        "target_date": target_date,
                        "metric": "high_temp_f",
                        "forecast_value_f": t_f,
                        "lead_hours": fhour,
                        "source_name": "NOAA",
                        "ingestion_time_utc": ingestion_ts,
                    }
                )

    logger.info(
        "Completed: %d rows collected, %d/%d skipped",
        len(records), skipped, total,
    )

    if not records:
        raise RuntimeError(
            f"No GFS forecast rows collected for {city} {year}-{month:02d}. "
            "All runs were skipped — check network connectivity and that "
            f"the AWS archive covers {year}-{month:02d}."
        )

    return pd.DataFrame(records, columns=_OUTPUT_COLUMNS)


def save_forecast_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write *df* to Parquet at *output_path*, creating parent dirs as needed.

    Args:
        df: DataFrame conforming to the ``forecast_temperature_point`` schema.
        output_path: Destination file path.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    Raises:
        ValueError: If *df* is empty.

    Example::

        out = save_forecast_parquet(df, "data/historical/forecasts/gfs_chicago_2023.parquet")
        print(f"Saved {len(df)} rows to {out}")
    """
    if df.empty:
        raise ValueError("Cannot save an empty DataFrame to Parquet.")

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing file so monthly runs accumulate rather than overwrite.
    if path.exists():
        existing = pd.read_parquet(path)
        df = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates(subset=["city", "model_name", "run_time_utc", "target_date", "metric"])
            .sort_values(["target_date", "run_time_utc"])
            .reset_index(drop=True)
        )

    df.to_parquet(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description=(
            "Fetch archival GFS 0.25° forecasts from NOAA AWS Open Data.\n\n"
            "Requires: pip install eccodes cfgrib xarray"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--city",   default="Chicago",
                   help="City name (must match cities.yaml)")
    p.add_argument("--year",   type=int, default=2023,
                   help="Year of the GFS archive month")
    p.add_argument("--month",  type=int, default=8,
                   help="Month (1-12) of the GFS archive month")
    p.add_argument("--cycles", default="00,12",
                   help="Comma-separated run cycles, e.g. '00,12'")
    p.add_argument("--leads",  default="1,2,3,4,5",
                   help="Comma-separated lead days, e.g. '1,2,3,4,5'")
    p.add_argument("--cache",
                   default="data/historical/forecasts/.grib_cache",
                   help="Directory for cached GRIB2 snippets")
    p.add_argument("--out",
                   default="data/historical/forecasts/gfs_chicago_2023.parquet",
                   help="Output Parquet file path")
    args = p.parse_args()

    try:
        df = fetch_gfs_forecasts(
            city=args.city,
            year=args.year,
            month=args.month,
            cycles=args.cycles.split(","),
            lead_days=[int(x) for x in args.leads.split(",")],
            cache_dir=args.cache,
        )
    except ImportError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path = save_forecast_parquet(df, args.out)

    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Lead hours: {sorted(df['lead_hours'].unique())}")
    print(f"Run cycles (UTC hour): {sorted(df['run_time_utc'].dt.hour.unique())}")
    print(f"Target date range: {df['target_date'].min()} to {df['target_date'].max()}")
    print()
    print(df.head(10).to_string(index=False))
