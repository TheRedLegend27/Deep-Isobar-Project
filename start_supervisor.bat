@echo off
:: Deep Isobar - daily-job supervisor (session / intraday / settle / dashboard)
:: Register once in Task Scheduler to run at logon under YOUR user account:
::   schtasks /create /sc onlogon /tn "DeepIsobarSupervisor" /tr "\"%~dp0start_supervisor.bat\"" /f
cd /d "%~dp0"
call "%~dp0.venv\Scripts\activate.bat"
python -m deep_isobar.supervisor
