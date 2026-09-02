@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0MixTeX-Latex-OCR\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo MixTeX virtual environment was not found:
  echo %PYTHON%
  pause
  exit /b 1
)
"%PYTHON%" "%~dp0table_recognizer_gui_mixtex.py"
if errorlevel 1 pause

