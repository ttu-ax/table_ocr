@echo off
rem PaddleOCR-VL-1.6 (vLLM 远程服务) 表格识别工具启动脚本
cd /d "%~dp0"
python table_recognizer_gui_paddleocr_vl.py
if errorlevel 1 pause
