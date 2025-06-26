@echo off
REM AetherCV Training Script for Windows
REM Usage: run_experiment.bat [config_name]

REM Set default config if none provided
if "%1"=="" (
    set CONFIG_NAME=single_yat_cnn
) else (
    set CONFIG_NAME=%1
)

echo 🚀 Starting AetherCV experiment with configuration: %CONFIG_NAME%
echo ============================================================

REM Check if config exists by listing available configs first
echo 📋 Available configurations:
python train.py --list-configs

echo.
echo 🎯 Running experiment with config: %CONFIG_NAME%
echo ============================================================

REM Run the experiment
python train.py --config %CONFIG_NAME%

echo.
echo ✅ Experiment completed!
echo ============================================================

pause
