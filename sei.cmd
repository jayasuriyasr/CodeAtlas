@echo off
rem SEI on this machine. One entry point, so it runs wherever PowerShell scripts
rem are blocked by policy:  sei setup | start <project> | stop | status | ask "..."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\sei.ps1" %*
