# Deep Isobar — PowerShell convenience wrapper (mirrors Makefile targets)
# Usage: .\run.ps1 <target>
param([string]$Target = "")

Set-Location $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot

switch ($Target) {

    # ── Trading ──────────────────────────────────────────────────────────────

    "dry" {
        python -m deep_isobar.research.paper_trade_session --dry-run
    }

    "trade" {
        python -m deep_isobar.research.paper_trade_session
    }

    "settle" {
        python -m deep_isobar.research.settle_paper_trades
    }

    # ── Calibration ──────────────────────────────────────────────────────────

    "replay" {
        python -m deep_isobar.calibration.historical_replay `
            --station KMDW `
            --start   2021-01-01 `
            --end     2024-12-31 `
            --workers 1
    }

    "onboard-dallas" {
        python -m deep_isobar.calibration.onboard_city `
            --city          Dallas `
            --station       KDFW `
            --lat           32.897 `
            --lon           -97.038 `
            --timezone      America/Chicago `
            --history-years 5
    }

    # ── Diagnostics ──────────────────────────────────────────────────────────

    "audit" {
        @'
import pandas as pd, pathlib, sys
base = pathlib.Path("data/bias_profiles")
if not base.exists():
    print("data/bias_profiles/ not found — run:  .\\run.bat replay"); sys.exit(0)
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
'@ | python
    }

    "logs" {
        $files = Get-ChildItem "data\paper_trades" -Include "*.log","*.txt","*.csv" `
                     -ErrorAction SilentlyContinue |
                 Sort-Object LastWriteTime -Descending
        if (-not $files) {
            Write-Host "No log files found in data\paper_trades\"
            exit 1
        }
        $latest = $files[0].FullName
        Write-Host "==> $latest"
        Get-Content $latest -Wait -Tail 100
    }

    "status" {
        Write-Host "=== Today ==="
        Get-Date -Format "dddd, MMMM dd yyyy  HH:mm:ss"

        Write-Host "`n=== Last paper_trades.csv row ==="
        $csv = "data\paper_trades\paper_trades.csv"
        if (Test-Path $csv) { Get-Content $csv -Tail 1 } else { Write-Host "(not found)" }

        Write-Host "`n=== Last raw_errors.parquet row ==="
        @'
import pandas as pd, pathlib, sys
f = next(pathlib.Path("data/bias_profiles").glob("*_raw_errors.parquet"), None)
if f is None:
    print("(not found)")
    sys.exit(0)
df = pd.read_parquet(f)
print(f"  file : {f.name}  ({len(df)} rows total)")
print(df.tail(1).to_string(index=False))
'@ | python

        Write-Host "`n=== API health ==="
        try {
            $r = Invoke-RestMethod -Uri "http://localhost:8765/api/health" -TimeoutSec 3
            $r | ConvertTo-Json -Depth 5
        } catch {
            Write-Host "(API not running)"
        }
    }

    "down" {
        foreach ($port in @(8765, 5173)) {
            Write-Host "Killing port $port ..."
            $pids = netstat -ano |
                    Select-String "\:$port\s" |
                    ForEach-Object { ($_.Line.Trim() -split '\s+')[-1] } |
                    Sort-Object -Unique
            foreach ($p in $pids) {
                if ($p -match '^\d+$' -and [int]$p -ne 0) {
                    Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue
                }
            }
        }
        Write-Host "Done."
    }

    # ── Help ─────────────────────────────────────────────────────────────────

    default {
        Write-Host ""
        Write-Host "  Usage: .\run.ps1 <target>   (or: run.bat <target>)"
        Write-Host ""
        Write-Host "  Trading"
        Write-Host "    dry             Score today's markets, print signals, place nothing"
        Write-Host "    trade           Live paper-trade session  ->  data/paper_trades/"
        Write-Host "    settle          Settle yesterday's contracts vs NOAA observed temps"
        Write-Host ""
        Write-Host "  Calibration"
        Write-Host "    replay          Re-run KMDW historical calibration (2021-2024)"
        Write-Host "    onboard-dallas  Onboard KDFW (Dallas) as a new city"
        Write-Host ""
        Write-Host "  Diagnostics"
        Write-Host "    audit           Parquet row counts + April bias values"
        Write-Host "    logs            Tail the most recent file in data/paper_trades/"
        Write-Host "    status          Date, last CSV row, last error row, API health"
        Write-Host "    down            Kill dev servers on ports 8765 and 5173"
        Write-Host ""
    }
}
