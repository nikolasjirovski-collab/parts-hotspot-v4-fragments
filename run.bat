@echo off
setlocal
cd /d "%~dp0"

echo Starting Parts Hotspot app v3...
echo Folder: %cd%
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Failed to create .venv. Check that Python is installed.
        pause
        exit /b 1
    )
)

if not exist ".venv\.deps-installed" (
    echo Installing dependencies. First launch can take several minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto deps_error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto deps_error
    type nul > ".venv\.deps-installed"
)

".venv\Scripts\python.exe" -c "import PIL, winsdk, pytesseract, fitz, cv2" >nul 2>nul
if errorlevel 1 (
    echo Dependencies are incomplete. Reinstalling...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto deps_error
)

echo Launching app...
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
    echo.
    echo App exited with an error.
    pause
    exit /b 1
)

exit /b 0

:deps_error
echo.
echo Failed to install dependencies.
pause
exit /b 1
