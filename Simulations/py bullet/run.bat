@echo off
REM ==============================================================================
REM PyBullet Quadrotor Simulation Launcher
REM Automatically uses the configured Python environment with PyBullet support.
REM ==============================================================================

python -c "import pybullet" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python main.py %*
) else (
    "C:\msys64\ucrt64\bin\python.exe" main.py %*
)

if "%~1"=="" pause
