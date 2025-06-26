#!/bin/bash

# Quick experiment scripts for AetherCV

echo "🚀 AetherCV Quick Experiment Launcher"
echo "====================================="

# Function to run experiment
run_experiment() {
    local config_name=$1
    local description=$2
    
    echo ""
    echo "🎯 $description"
    echo "Configuration: $config_name"
    echo "-------------------------------------"
    python train.py --config "$config_name"
}

# Parse command line argument
case $1 in
    "single"|"1")
        run_experiment "single_yat_cnn" "Training single YatCNN model"
        ;;
    "full"|"all"|"2")
        run_experiment "full_comparison" "Full 4-model comparison"
        ;;
    "cnn"|"3")
        run_experiment "cnn_stl10" "CNN models on STL10"
        ;;
    "resnet"|"4")
        run_experiment "resnet_comparison" "ResNet models comparison"
        ;;
    "eurosat"|"satellite"|"5")
        run_experiment "eurosat_analysis" "Satellite imagery analysis"
        ;;
    "list"|"ls")
        echo "📋 Listing all available configurations:"
        python train.py --list-configs
        ;;
    *)
        echo "Usage: $0 [experiment_type]"
        echo ""
        echo "Available experiment types:"
        echo "  single, 1     - Train single YatCNN model"
        echo "  full, all, 2  - Compare all 4 models"
        echo "  cnn, 3        - Compare CNN models on STL10"
        echo "  resnet, 4     - Compare ResNet models"
        echo "  eurosat, 5    - Satellite imagery analysis"
        echo "  list, ls      - List all available configs"
        echo ""
        echo "Examples:"
        echo "  $0 single"
        echo "  $0 full"
        echo "  $0 cnn"
        echo "  $0 list"
        ;;
esac
