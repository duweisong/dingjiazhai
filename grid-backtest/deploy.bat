@echo off
REM ============================================================
REM  网格交易系统 — Windows 定时任务部署脚本
REM  以管理员身份运行此脚本
REM ============================================================

set SCRIPT_DIR=C:\AI\grid-backtest
set PYTHON=C:\Program Files\Python312\python.exe

echo.
echo ========================================
echo   网格交易系统 — 部署定时任务
echo ========================================
echo.

REM 1. 创建每日信号任务 (每个交易日 15:30)
schtasks /create /tn "GridSignal_Daily" ^
  /tr "\"%PYTHON%\" \"%SCRIPT_DIR%\run_daily_signal.py\" --push" ^
  /sc weekly /d MON,TUE,WED,THU,FRI /st 15:30 ^
  /f

echo [OK] 每日信号任务已创建 (周一至周五 15:30)

REM 2. 创建每周报告任务 (每周五 16:00)
schtasks /create /tn "GridReport_Weekly" ^
  /tr "\"%PYTHON%\" \"%SCRIPT_DIR%\run_batch_backtest.py\" --preset moderate" ^
  /sc weekly /d FRI /st 16:00 ^
  /f

echo [OK] 每周报告任务已创建 (周五 16:00)

echo.
echo ========================================
echo   部署完成！
echo   查看任务: schtasks /query | findstr Grid
echo   删除任务: schtasks /delete /tn "任务名" /f
echo ========================================
echo.
pause
