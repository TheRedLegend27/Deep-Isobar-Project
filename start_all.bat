@echo off
:: Deep Isobar — launch API and dashboard in separate console windows
echo Launching Deep Isobar API ...
start "Deep Isobar API" cmd /k "%~dp0start_api.bat"

echo Launching Deep Isobar dashboard ...
start "Deep Isobar UI" cmd /k "%~dp0start_dash.bat"

echo.
echo  API  ^> http://localhost:8765/api/health
echo  UI   ^> http://localhost:5173
echo.
echo Both servers are running in separate windows.
