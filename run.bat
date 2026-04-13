@echo off
:: Deep Isobar — Windows convenience wrapper (mirrors Makefile targets)
:: Usage: run <target>
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
