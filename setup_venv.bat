@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo === Cai dat moi truong (chi chay lan dau) ===
echo.

if exist "venv_win\Scripts\python.exe" (
    echo [Info] venv_win da ton tai. Bo qua tao venv.
    set "PYTHON_EXE=venv_win\Scripts\python.exe"
    goto install_deps
)

echo [Info] Tao venv_win...
python -m venv venv_win
if errorlevel 1 (
    echo [Error] Khong tao duoc venv. Dong server/CMD dang dung python.exe roi thu lai.
    pause
    exit /b 1
)
set "PYTHON_EXE=venv_win\Scripts\python.exe"

:install_deps
if not exist ".env" (
    echo [Info] Tao file .env tu .env.example...
    copy /Y ".env.example" ".env" >nul
    echo [Info] Mo .env va dien it nhat 1 API key ^(GROQ_API_KEY, GEMINI_API_KEY, ...^)
)

echo [Info] Cai dependencies...
"%PYTHON_EXE%" -m pip install --upgrade pip
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [Error] pip install that bai.
    pause
    exit /b 1
)

echo.
echo [OK] Xong. Khoi dong app:
echo   - Double-click: MO_APP.bat
echo   - PowerShell:   .\run_app.bat
pause
endlocal
