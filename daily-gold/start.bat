@echo off
cd /d C:\AI\daily-gold
echo ======================================
echo   掘金日报 Pro - 启动服务器
echo ======================================
echo.
if not exist node_modules (
  echo [首次运行] 安装依赖...
  call npm install
)
echo.
echo 启动服务器...
node server/server.js
pause
