@echo off
REM One-click start for Windows.
REM First time: creates a virtual environment (.venv) and installs requirements.txt.
REM Every time: runs the car. Extra arguments are passed on, e.g.  run.bat --compare
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment .venv ...
    python -m venv .venv || py -m venv .venv
)
echo Installing requirements (fast after the first time) ...
.venv\Scripts\python -m pip install -q -r requirements.txt
.venv\Scripts\python lane_car.py %*
pause
