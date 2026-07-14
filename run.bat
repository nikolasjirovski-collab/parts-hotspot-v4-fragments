@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo Starting Parts Hotspot OCR v4.20...
echo.

set "BOOTSTRAP_PYTHON="

if not exist ".venv\Scripts\python.exe" (
    call :find_supported_python
    if not defined BOOTSTRAP_PYTHON call :install_python
    if not defined BOOTSTRAP_PYTHON goto python_error

    echo Creating local Python environment...
    "%BOOTSTRAP_PYTHON%" -m venv .venv
    if errorlevel 1 goto venv_error
)

if not exist ".venv\.deps-installed" (
    echo Installing application dependencies. The first launch can take several minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto deps_error
    ".venv\Scripts\python.exe" -m pip install --force-reinstall torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto deps_error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto deps_error
    type nul > ".venv\.deps-installed"
)

".venv\Scripts\python.exe" -c "import PIL, winsdk, pytesseract, fitz, cv2, ultralytics, rapidocr, onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo Repairing incomplete dependencies...
    ".venv\Scripts\python.exe" -m pip install --force-reinstall torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto deps_error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto deps_error
)

echo Launching application...
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto app_error
exit /b 0

:find_supported_python
for %%V in (3.12 3.11 3.10) do (
    for /f "usebackq delims=" %%P in (`py -%%V -c "import sys; print(sys.executable)" 2^>nul`) do (
        set "BOOTSTRAP_PYTHON=%%P"
        goto :eof
    )
)
for /f "usebackq delims=" %%P in (`python -c "import sys; assert (3, 10) ^<= sys.version_info[:2] ^<= (3, 12); print(sys.executable)" 2^>nul`) do (
    set "BOOTSTRAP_PYTHON=%%P"
    goto :eof
)
goto :eof

:install_python
where winget >nul 2>nul
if errorlevel 1 goto :eof
echo Python 3.10-3.12 was not found. Installing Python 3.12 for this user...
winget install --id Python.Python.3.12 --exact --scope user --accept-package-agreements --accept-source-agreements --silent
if errorlevel 1 goto :eof
call :find_supported_python
goto :eof

:python_error
echo.
echo Python 3.10, 3.11, or 3.12 could not be installed automatically.
echo Install Python 3.12 from https://www.python.org/downloads/ and run this file again.
pause
exit /b 1

:venv_error
echo.
echo Failed to create the local Python environment.
pause
exit /b 1

:deps_error
echo.
echo Failed to install application dependencies. Check the Internet connection and run this file again.
pause
exit /b 1

:app_error
echo.
echo Application exited with an error.
pause
exit /b 1
