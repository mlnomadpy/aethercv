@echo off
echo AetherCV WandB Logging Setup and Test
echo =====================================

echo.
echo Step 1: Installing WandB if not already installed...
pip install wandb

echo.
echo Step 2: Testing logging system...
python test_wandb_logging.py

echo.
echo Step 3: Instructions for using WandB logging:
echo.
echo 1. Login to WandB (run this once):
echo    wandb login
echo.
echo 2. Update your configuration in config/logging_config.py
echo.
echo 3. Run example training with logging:
echo    python example_wandb_usage.py --example single
echo.
echo 4. View results at: https://wandb.ai/
echo.
pause
