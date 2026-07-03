"""Nightly backup of the unreplaceable ``data/`` assets.

Code lives in git and EMOS params refit in minutes — but the trade log,
the accumulated ensemble-spread history, training parquets, and the
orderbook archive cannot be regenerated.  (Lesson of 2026-07-02, when
iCloud eviction destroyed the working tree.)

Creates ``deep_isobar_data_YYYYMMDD_HHMMSS.tar.gz`` in ``backup.dest_dir``
(default ``~/DeepIsobarBackups`` — keep it outside the repo and any cloud-
synced path), rotates old archives beyond ``backup.keep``, and optionally
pushes a copy to an rclone remote when ``backup.rclone_remote`` is set
(Phase 1: retarget to TrueNAS + offsite).

Included: paper_trades, emos_params, emos_training, bias_profiles,
reports, market_history, scheduler_state.json.
Excluded: logs and download caches (regenerable bulk).

CLI::

    python -m deep_isobar.ops.backup
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

from deep_isobar.config import get_setting

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"

# Directories/files under data/ worth protecting.  Anything not listed is
# treated as regenerable (caches, logs).
_INCLUDE = [
    "paper_trades",
    "emos_params",
    "emos_training",
    "bias_profiles",
    "reports",
    "market_history",
    "scheduler_state.json",
]


def run_backup(
    dest_dir: Path | None = None,
    keep: int | None = None,
    data_dir: Path | None = None,
) -> Path | None:
    """Create one rotated archive; returns its path (None if nothing to back up)."""
    data = data_dir or _DATA_DIR
    dest = Path(
        dest_dir
        or Path(str(get_setting("backup.dest_dir", default="~/DeepIsobarBackups")))
    ).expanduser()
    keep = keep if keep is not None else int(get_setting("backup.keep", default=14))

    sources = [(data / name) for name in _INCLUDE if (data / name).exists()]
    if not sources:
        logger.warning("backup: nothing to back up under %s", data)
        return None

    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = dest / f"deep_isobar_data_{stamp}.tar.gz"
    n = 1
    while archive.exists():  # same-second rerun must not overwrite
        archive = dest / f"deep_isobar_data_{stamp}_{n}.tar.gz"
        n += 1

    with tarfile.open(archive, "w:gz") as tar:
        for src in sources:
            tar.add(src, arcname=f"data/{src.name}")
    size_mb = archive.stat().st_size / 1e6
    logger.info("backup: wrote %s (%.1f MB, %d top-level items)",
                archive, size_mb, len(sources))

    # ── Rotation ─────────────────────────────────────────────────────────
    archives = sorted(dest.glob("deep_isobar_data_*.tar.gz"))
    for old in archives[:-keep] if keep > 0 else []:
        old.unlink()
        logger.info("backup: rotated out %s", old.name)

    # ── Optional offsite copy via rclone ─────────────────────────────────
    remote = get_setting("backup.rclone_remote", default=None)
    if remote:
        rclone = shutil.which("rclone")
        if rclone is None:
            logger.warning("backup: rclone_remote set but rclone not installed")
        else:
            try:
                subprocess.run(
                    [rclone, "copy", str(archive), str(remote)],
                    check=True, timeout=600, capture_output=True,
                )
                logger.info("backup: offsite copy → %s", remote)
            except Exception as exc:  # noqa: BLE001
                logger.error("backup: offsite copy failed: %s", exc)

    return archive


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return 0 if run_backup() is not None else 1


if __name__ == "__main__":
    sys.exit(main())
