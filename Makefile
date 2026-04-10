# Deep Isobar — developer convenience targets
# Requires: GNU Make, Python 3.11+, Node 18+
# On Windows use WSL, Git Bash, or the start_*.bat equivalents instead.

SHELL := /bin/bash

.PHONY: api dash up down trade settle dry replay onboard-dallas audit logs status

# ── Servers ──────────────────────────────────────────────────────────────────

api:
	PYTHONPATH=. uvicorn deep_isobar.dashboard.api:app \
	    --reload --host 0.0.0.0 --port 8765

dash:
	cd dashboard_ui && npm run dev

up:
	@echo "Starting API on http://localhost:8765 ..."
	@PYTHONPATH=. uvicorn deep_isobar.dashboard.api:app \
	    --reload --host 0.0.0.0 --port 8765 \
	    > /tmp/deep_isobar_api.log 2>&1 &
	@echo "Starting Vite dev server on http://localhost:5173 ..."
	@cd dashboard_ui && npm run dev \
	    > /tmp/deep_isobar_dash.log 2>&1 &
	@echo ""
	@echo "  API  → http://localhost:8765/api/health"
	@echo "  UI   → http://localhost:5173"
	@echo ""
	@echo "Tailing both logs (Ctrl-C to stop tailing — servers keep running):"
	@sleep 1
	@tail -f /tmp/deep_isobar_api.log /tmp/deep_isobar_dash.log

down:
	@echo "Killing port 8765 (API) ..."
	@-lsof -ti :8765 | xargs kill -9 2>/dev/null || true
	@echo "Killing Vite dev server (port 5173) ..."
	@-lsof -ti :5173 | xargs kill -9 2>/dev/null || true
	@echo "Done."

# ── Trading ───────────────────────────────────────────────────────────────────

trade:
	PYTHONPATH=. python -m deep_isobar.research.paper_trade_session

settle:
	PYTHONPATH=. python -m deep_isobar.research.settle_paper_trades

dry:
	PYTHONPATH=. python -m deep_isobar.research.paper_trade_session --dry-run

# ── Data / calibration ────────────────────────────────────────────────────────

replay:
	PYTHONPATH=. python -m deep_isobar.calibration.historical_replay \
	    --station KMDW \
	    --start 2021-01-01 \
	    --end   2024-12-31 \
	    --workers 1

onboard-dallas:
	PYTHONPATH=. python -m deep_isobar.calibration.onboard_city \
	    --city Dallas \
	    --station KDFW \
	    --lat 32.897 \
	    --lon -97.038 \
	    --timezone America/Chicago \
	    --history-years 5

# ── Diagnostics ───────────────────────────────────────────────────────────────

audit:
	@python - <<'EOF'
import pandas as pd, pathlib, sys
base = pathlib.Path("data/bias_profiles")
if not base.exists():
    print("data/bias_profiles/ not found — run `make replay` first"); sys.exit(0)
for f in sorted(base.glob("*.parquet")):
    df = pd.read_parquet(f)
    print(f"\n{f.name}  ({len(df)} rows)")
    if "month" in df.columns:
        apr = df[df["month"] == 4]
        if not apr.empty:
            print("  April rows:")
            print(apr.to_string(index=False))
        else:
            print("  (no April rows)")
    else:
        print(df.head(5).to_string(index=False))
EOF

logs:
	@latest=$$(ls -t data/paper_trades/*.{log,txt,csv} 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then echo "No log files found in data/paper_trades/"; exit 1; fi; \
	echo "==> $$latest"; \
	tail -f -n 100 "$$latest"

status:
	@echo "=== Today ==="
	@date
	@echo ""
	@echo "=== Last paper_trades.csv row ==="
	@tail -1 data/paper_trades/paper_trades.csv 2>/dev/null || echo "(not found)"
	@echo ""
	@echo "=== Last raw_errors.parquet row ==="
	@python - <<'EOF'
import pandas as pd, pathlib, sys
f = next(pathlib.Path("data/bias_profiles").glob("*_raw_errors.parquet"), None)
if f is None:
    print("(not found)")
    sys.exit(0)
df = pd.read_parquet(f)
print(f"  file : {f.name}  ({len(df)} rows total)")
print(df.tail(1).to_string(index=False))
EOF
	@echo ""
	@echo "=== API health ==="
	@curl -s http://localhost:8765/api/health 2>/dev/null || echo "(API not running)"
