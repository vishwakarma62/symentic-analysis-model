@echo off
REM Quick APK Analysis Runner for Windows
REM Usage: run_analysis.bat [apk_path]

echo ========================================
echo    Homitra APK Analysis Tool
echo ========================================
echo.

if "%1"=="" (
    echo Running analysis on default APK...
    powershell -ExecutionPolicy Bypass -File "%~dp0analyze_apk.ps1"
) else (
    echo Running analysis on: %1
    powershell -ExecutionPolicy Bypass -File "%~dp0analyze_apk.ps1" -ApkPath "%1"
)

echo.
echo Analysis complete!
pause