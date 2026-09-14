@echo off
echo.
echo  ===================================================
echo   AudioShield Forensics - Web Interface
echo  ===================================================
echo.
echo  Starting server at http://localhost:8000
echo  Press Ctrl+C to stop
echo.

call venv\Scripts\activate
python -m src.api.run

pause
