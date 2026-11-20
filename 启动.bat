@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set HF_ENDPOINT=https://hf-mirror.com

set PYEXE="C:\Users\Admin\miniconda3\envs\two\python.exe"
if not exist %PYEXE% set PYEXE=python

echo.
echo   ============================================
echo     法律条文问答  LegalKB-Agent
echo   ============================================
echo.
echo   模型: 本地 bge 语义向量 + qwen3.8-omni-flash
echo   地址: http://127.0.0.1:8000
echo.
echo   关闭本窗口即停止服务
echo.

start "LegalKB-Agent" /min cmd /c "%PYEXE% -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

timeout /t 6 >nul
start "" http://127.0.0.1:8000

echo   浏览器已打开。如果显示"无法访问"，等 3 秒刷新一次（首次要加载模型）。
echo.
pause >nul
