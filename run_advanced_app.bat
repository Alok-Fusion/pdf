@echo off
echo ===================================================
echo   Advanced Mechanical PDF Analyzer - Launcher
echo ===================================================

cd /d "%~dp0"

IF EXIST ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
) ELSE (
    set PYTHON_CMD=python
)

echo [INFO] Launching Advanced Extraction App...
"%PYTHON_CMD%" -m streamlit run advanced_extraction/app_advanced.py
pause
