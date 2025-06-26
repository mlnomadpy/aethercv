@echo off
REM Quick experiment scripts for AetherCV

echo 🚀 AetherCV Quick Experiment Launcher
echo =====================================

REM Parse command line argument
if "%1"=="single" goto single
if "%1"=="1" goto single
if "%1"=="full" goto full
if "%1"=="all" goto full
if "%1"=="2" goto full
if "%1"=="cnn" goto cnn
if "%1"=="3" goto cnn
if "%1"=="resnet" goto resnet
if "%1"=="4" goto resnet
if "%1"=="eurosat" goto eurosat
if "%1"=="satellite" goto eurosat
if "%1"=="5" goto eurosat
if "%1"=="list" goto list
if "%1"=="ls" goto list
goto usage

:single
echo.
echo 🎯 Training single YatCNN model
echo Configuration: single_yat_cnn
echo -------------------------------------
python train.py --config single_yat_cnn
goto end

:full
echo.
echo 🎯 Full 4-model comparison
echo Configuration: full_comparison
echo -------------------------------------
python train.py --config full_comparison
goto end

:cnn
echo.
echo 🎯 CNN models on STL10
echo Configuration: cnn_stl10
echo -------------------------------------
python train.py --config cnn_stl10
goto end

:resnet
echo.
echo 🎯 ResNet models comparison
echo Configuration: resnet_comparison
echo -------------------------------------
python train.py --config resnet_comparison
goto end

:eurosat
echo.
echo 🎯 Satellite imagery analysis
echo Configuration: eurosat_analysis
echo -------------------------------------
python train.py --config eurosat_analysis
goto end

:list
echo 📋 Listing all available configurations:
python train.py --list-configs
goto end

:usage
echo Usage: %0 [experiment_type]
echo.
echo Available experiment types:
echo   single, 1     - Train single YatCNN model
echo   full, all, 2  - Compare all 4 models
echo   cnn, 3        - Compare CNN models on STL10
echo   resnet, 4     - Compare ResNet models
echo   eurosat, 5    - Satellite imagery analysis
echo   list, ls      - List all available configs
echo.
echo Examples:
echo   %0 single
echo   %0 full
echo   %0 cnn
echo   %0 list

:end
pause
