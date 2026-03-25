@echo off
cd /d %~dp0

REM --- Activate venv ---
call "%~dp0.venv\Scripts\activate.bat"

REM --- Start backend ---
start "Backend" cmd /k python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000

REM --- Give backend time to start ---
timeout /t 3 >nul

REM --- Start frontend ---
start "Frontend" cmd /k python -m streamlit run app.py
