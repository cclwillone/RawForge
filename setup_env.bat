@echo off
rem RawForge 一键安装脚本 (Windows 10/11)
rem 1. 使用当前 python 创建虚拟环境 .venv
rem 2. 安装依赖 (优先清华镜像加速)
chcp 65001 >nul
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python, 请先安装 Python 3.9+ 并勾选 "Add to PATH"
    pause
    exit /b 1
)

echo [1/3] 创建虚拟环境 .venv ...
if not exist ".venv" (
    python -m venv .venv
)

echo [2/3] 升级 pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip -q

echo [3/3] 安装依赖 (清华镜像) ...
".venv\Scripts\python.exe" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

echo.
echo 安装完成! 启动程序:
echo   双击 run.bat  或  运行 .venv\Scripts\python.exe main.py
pause
