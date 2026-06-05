@echo off
setlocal EnableExtensions
title NBLM Research
cd /d "%~dp0"
set PYTHONUTF8=1

echo.
echo  =============================================
echo       NBLM Research - Dang khoi dong...
echo  =============================================
echo.
echo  Thu muc: %CD%
echo.

REM --- Tim Python (uu tien venv_win vi project dung no) ---
set "PYTHON_EXE="
if exist "venv_win\Scripts\python.exe" set "PYTHON_EXE=%CD%\venv_win\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
if not defined PYTHON_EXE (
    where python >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo  [LOI] Khong tim thay Python.
    echo  Chay file: setup_venv.bat
    echo.
    pause
    exit /b 1
)

if not exist "app\main.py" (
    echo  [LOI] Khong thay app\main.py - hay mo .bat trong thu muc doan.
    pause
    exit /b 1
)

echo  [OK] Python: %PYTHON_EXE%
echo.

REM --- gRPC Vision (chi can khi SCREEN_CAPTURE_ENABLED=true trong .env) ---
set "SKIP_GRPC=0"
if exist ".env" findstr /I /C:"SCREEN_CAPTURE_ENABLED=false" ".env" >nul && set "SKIP_GRPC=1"
if "%SKIP_GRPC%"=="1" (
    echo  [Info] Screen capture TAT - bo qua gRPC Vision ^(chay nhe hon^)
) else (
    set "GRPC_RUNNING=0"
    for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":50051" ^| findstr "LISTENING"') do set "GRPC_RUNNING=1"
    if "%GRPC_RUNNING%"=="1" (
        echo  [Info] gRPC Vision: port 50051 OK
    ) else (
        echo  [Info] Khoi dong gRPC Vision ^(OCR man hinh^)...
        start "NBLM Vision" /MIN "%PYTHON_EXE%" "%CD%\app\workers\vision_grpc_server.py"
        ping 127.0.0.1 -n 4 >nul
    )
)

REM --- Giai phong port 8000 ---
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo  [Info] Dang giai phong port 8000 ^(PID %%P^)...
    taskkill /F /PID %%P >nul 2>&1
)
ping 127.0.0.1 -n 2 >nul

echo  [Info] Dang tai model AI ^(lan dau co the 30-90 giay^)...
echo  [Info] Server: http://localhost:8000
echo  [Info] Dung server: Ctrl+C trong cua so nay
echo.

REM Mo trinh duyet khi server san sang (khong mo som gay loi)
start "" /MIN cmd /c "powershell -NoProfile -Command "$u='http://127.0.0.1:8000/api/status'; for($i=0;$i -lt 120;$i++){ try { if((Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200){ Start-Process 'http://localhost:8000'; exit 0 } } catch {}; Start-Sleep -Seconds 2 }""

"%PYTHON_EXE%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo  [LOI] Server dung ^(ma loi %EXIT_CODE%^).
    echo  Thu chay: setup_venv.bat
    echo  Hoac: "%PYTHON_EXE%" -m pip install -r requirements.txt
)
echo.
pause
endlocal
exit /b %EXIT_CODE%
