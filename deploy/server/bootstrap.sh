#!/usr/bin/env bash
# Deep Isobar server bootstrap — Ubuntu 24.04 VM (Proxmox), idempotent.
#
#   sudo bash bootstrap.sh --role worker      # R520 backfill/sweep worker
#   sudo bash bootstrap.sh --role coordinator # R750: queue seeding, DuckDB, Grafana host
#
# Roles differ only in which systemd units are enabled; both get the full
# checkout, venv, and GRIB stack.  Re-running upgrades in place.
#
# Prereqs done by hand first (see docs/RUNBOOK_SERVERS.md):
#   1. Tailscale joined:      curl -fsSL https://tailscale.com/install.sh | sh && tailscale up
#   2. Data lake mounted:     /mnt/lake (NFS from TrueNAS; see runbook §3)
#   3. A GitHub deploy key or HTTPS token that can clone the repo.

set -euo pipefail

ROLE="worker"
REPO_URL="${DEEP_ISOBAR_REPO:-https://github.com/TheRedLegend27/Deep-Isobar-Project.git}"
APP_DIR="/opt/deep-isobar"
APP_USER="isobar"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;;
    --repo) REPO_URL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "== Deep Isobar bootstrap: role=$ROLE =="

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -qq
# libeccodes: GRIB decode for cfgrib (no file-locking issues on Linux).
apt-get install -y -qq git curl libeccodes0 libeccodes-data nfs-common

# ── App user + checkout ──────────────────────────────────────────────────────
id -u "$APP_USER" &>/dev/null || useradd -r -m -s /bin/bash "$APP_USER"
if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$APP_DIR"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# ── Python env (uv) ──────────────────────────────────────────────────────────
sudo -u "$APP_USER" bash -c '
  set -e
  command -v ~/.local/bin/uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  cd '"$APP_DIR"'
  ~/.local/bin/uv venv --allow-existing
  ~/.local/bin/uv pip install -e . --python .venv/bin/python
  ~/.local/bin/uv pip install cfgrib xarray eccodes --python .venv/bin/python
'

# ── Env file (never clobber an existing one) ────────────────────────────────
if [[ ! -f "$APP_DIR/.env" ]]; then
  sudo -u "$APP_USER" cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo "!! Edit $APP_DIR/.env (Kalshi creds only needed on the runtime role)"
fi

# ── systemd units per role ───────────────────────────────────────────────────
install -m 644 "$APP_DIR/deploy/server/deep-isobar-backfill.service" /etc/systemd/system/
install -m 644 "$APP_DIR/deploy/server/deep-isobar-backfill.timer" /etc/systemd/system/
systemctl daemon-reload

case "$ROLE" in
  worker)
    systemctl enable --now deep-isobar-backfill.timer
    echo "== worker ready: backfill timer enabled (runs now + hourly retry) =="
    ;;
  coordinator)
    echo "== coordinator ready. Next (manual, once): =="
    echo "   sudo -u $APP_USER $APP_DIR/.venv/bin/python -m deep_isobar.lab.gefs_backfill \\"
    echo "     --seed --start 2021-01 --end 2026-06 \\"
    echo "     --queue /mnt/lake/backfill/queue --out /mnt/lake/backfill/out"
    echo "   (Grafana/DuckDB setup: runbook §6)"
    ;;
  *) echo "unknown role: $ROLE" >&2; exit 2 ;;
esac

echo "== bootstrap complete =="
