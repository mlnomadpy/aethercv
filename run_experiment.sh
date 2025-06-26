#!/bin/bash

# AetherCV Training Script
# Usage: ./run_experiment.sh [config_name]

# Set default config if none provided
CONFIG_NAME=${1:-"single_yat_cnn"}

echo "🚀 Starting AetherCV experiment with configuration: $CONFIG_NAME"
echo "============================================================"

# Check if config exists by listing available configs first
echo "📋 Available configurations:"
python train.py --list-configs

echo ""
echo "🎯 Running experiment with config: $CONFIG_NAME"
echo "============================================================"

# Run the experiment
python train.py --config "$CONFIG_NAME"

echo ""
echo "✅ Experiment completed!"
echo "============================================================"
