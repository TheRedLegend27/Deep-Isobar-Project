@echo off
:: Deep Isobar — start FastAPI backend on port 8765
SET PYTHONPATH=%~dp0
cd /d "%~dp0"
echo Starting Deep Isobar API on http://localhost:8765 ...
uvicorn deep_isobar.dashboard.api:app --reload --host 0.0.0.0 --port 8765
