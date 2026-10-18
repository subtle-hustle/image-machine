@echo off
setlocal
title Image Machine
cd /d "%~dp0"
set "UV_INSTALL_DIR=%~dp0tools"
set "UV_CACHE_DIR=%~dp0tools\cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0tools\python"
set "UV_NO_MODIFY_PATH=1"
set "UV=%~dp0tools\uv.exe"

if exist ".venv\installed.txt" goto run

echo.
echo  First-time setup: downloading Python, PyTorch and the app's parts.
echo  This takes a while (about 3 GB) and only happens once.
echo.
if not exist "%UV%" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || goto fail
)
"%UV%" venv --python 3.11 .venv || goto fail
"%UV%" pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126 || goto fail
"%UV%" pip install --python .venv\Scripts\python.exe -r requirements.txt || goto fail
echo ok> ".venv\installed.txt"

:run
echo.
echo  Starting Image Machine. Your browser will open in a moment.
echo  Keep this window open while you use it; close it to quit.
echo.
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto fail
goto :eof

:fail
echo.
echo  Something went wrong. Take a screenshot of this window and send it to Claude.
pause
