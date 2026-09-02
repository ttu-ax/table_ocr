@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo 依赖安装失败，请检查 Python 和网络设置。
  pause
  exit /b 1
)
py -3 table_recognizer_gui.py
if errorlevel 1 pause
