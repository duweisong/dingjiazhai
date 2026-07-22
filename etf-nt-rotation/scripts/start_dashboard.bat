@echo off
echo ========================================
echo   ETF NT Rotation - API Server
echo ========================================
echo.
echo   Starting API server on port 8765...
echo   Then open: C:\AI\etf-nt-rotation\dashboard.html
echo   Press Ctrl+C to stop
echo.
cd /d C:\AI
python C:\AI\etf-nt-rotation\scripts\dashboard_server.py
pause
