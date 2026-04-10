@echo off
:: Deep Isobar — start React / Vite frontend on port 5173
cd /d "%~dp0dashboard_ui"
echo Starting Deep Isobar dashboard on http://localhost:5173 ...
npm run dev
