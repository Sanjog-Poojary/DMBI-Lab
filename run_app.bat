@echo off
title DMBI Video Streaming BI Dashboard
cd /d "%~dp0"

echo ========================================================
echo   DMBI Video Streaming Platform BI Dashboard (Roll 40)
echo ========================================================
echo.
echo Starting Streamlit app from: %CD%
echo Default browser will open at: http://localhost:8501
echo.

python -m streamlit run CA2_Mini_Project\app\app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application failed to start or exited with an error.
    echo Press any key to close this window...
    pause >nul
)

