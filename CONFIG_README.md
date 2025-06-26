# AetherCV Configuration System

This document describes the YAML-based configuration system for AetherCV experiments.

## Overview

AetherCV now supports both command-line arguments and YAML configuration files for running experiments. The YAML approach is recommended for reproducible experiments and complex configurations.

## Quick Start

### 1. List Available Configurations
```bash
python train.py --list-configs
```

### 2. Run an Experiment with YAML Config
```bash
python train.py --config single_yat_cnn
```

### 3. Use Convenience Scripts
```bash
# Linux/Mac
./quick_run.sh single
./quick_run.sh full
./quick_run.sh cnn

# Windows
quick_run.bat single
quick_run.bat full
quick_run.bat cnn
```

## Available Configurations

| Config File | Description | Models | Dataset | Analysis |
|-------------|-------------|---------|---------|----------|
| `single_yat_cnn` | Train only YatCNN | YatCNN | CIFAR-10 | Standard |
| `full_comparison` | Compare all 4 models | All 4 | CIFAR-10 | Comprehensive |
| `cnn_stl10` | CNN comparison on STL10 | YatCNN, LinearCNN | STL10 | Advanced |
| `resnet_comparison` | ResNet models | YatResNet, LinearResNet | CIFAR-100 | Standard |
| `eurosat_analysis` | Satellite imagery | YatCNN, YatResNet | EuroSAT RGB | Advanced |

## Configuration File Format

YAML configuration files are located in the `configs/` directory. Here's the structure:

```yaml
# Experiment metadata
experiment:
  name: "experiment_name"
  description: "Experiment description"

# Dataset configuration
dataset:
  name: "cifar10"          # Dataset name
  num_epochs: 5            # Number of training epochs
  eval_every: 200          # Evaluation frequency
  batch_size: 64           # Batch size (multiplied by device count)

# Models to train
models:
  - model_class_name: "YatCNN"
    display_name: "YatCNN"
    rng_seed: 0
    kernel_viz_layer: "conv1"
    activation_viz_layer: "conv1"
  - model_class_name: "LinearCNN"
    display_name: "LinearCNN"
    rng_seed: 0
    kernel_viz_layer: "conv1"
    activation_viz_layer: "conv1"

# Training parameters
training:
  learning_rate: 0.007
  optimizer: "adamw"       # adamw, adam, or sgd

# Analysis configuration
analysis:
  mode: "standard"         # basic, standard, advanced, comprehensive
  save_visualizations: true

# Output configuration
output:
  save_dir: null           # Auto-generate if null
  save_models: true
  save_metrics: true
```

## Creating Custom Configurations

1. Create a new YAML file in the `configs/` directory
2. Follow the format above
3. Use meaningful names and descriptions
4. Test your configuration:

```bash
python train.py --config your_config_name
```

## Supported Options

### Datasets
- `cifar10`, `cifar100`
- `stl10`
- `eurosat/rgb`, `eurosat/all`

### Models
- `YatCNN` - YAT-based CNN
- `LinearCNN` - Linear CNN
- `YatResNet` - YAT-based ResNet
- `LinearResNet` - Linear ResNet

### Analysis Modes
- `basic` - Training curves and basic metrics
- `standard` - Basic + activation maps, saliency, confusion matrices
- `advanced` - Standard + explainability methods
- `comprehensive` - Advanced + robustness and efficiency analysis

### Optimizers
- `adamw` - AdamW optimizer (recommended)
- `adam` - Adam optimizer
- `sgd` - Stochastic Gradient Descent

## Shell Scripts

### Basic Experiment Runner
```bash
# Run specific config
./run_experiment.sh single_yat_cnn
./run_experiment.sh full_comparison

# Windows
run_experiment.bat single_yat_cnn
run_experiment.bat full_comparison
```

### Quick Launcher
```bash
# Predefined experiments
./quick_run.sh single    # Single YatCNN
./quick_run.sh full      # All 4 models
./quick_run.sh cnn       # CNN comparison
./quick_run.sh resnet    # ResNet comparison
./quick_run.sh eurosat   # Satellite imagery

# Windows
quick_run.bat single
quick_run.bat full
quick_run.bat cnn
```

## Legacy Command-Line Interface

The old command-line interface is still supported:

```bash
# Single model
python train.py --model yat_cnn --dataset cifar10 --analysis-mode basic

# Model groups
python train.py --model-group cnn --dataset stl10 --analysis-mode standard
```

## Programmatic API

### Using YAML Configs
```python
from utils.yaml_config import ConfigLoader
from train import run_experiment_from_config

# Load configuration
loader = ConfigLoader()
config = loader.load_config('single_yat_cnn')

# Run experiment
results = run_experiment_from_config(config)
```

### Legacy API
```python
from train import run_complete_comparison, get_yat_cnn_only

# Run single model
results = run_complete_comparison(
    dataset_name='cifar10',
    model_configs_to_run=get_yat_cnn_only(),
    analysis_mode='standard'
)
```

## Best Practices

1. **Use YAML configs** for reproducible experiments
2. **Meaningful names** for experiment identification
3. **Version control** your configuration files
4. **Test configurations** before long runs
5. **Document experiments** with clear descriptions

## Troubleshooting

### Common Issues

1. **Config not found**: Use `--list-configs` to see available options
2. **Invalid YAML**: Check syntax with a YAML validator
3. **Model errors**: Ensure model class names are correct
4. **Dataset errors**: Verify dataset names are supported

### Getting Help

```bash
# List all configs
python train.py --list-configs

# Show help
python train.py --help

# Quick experiments
./quick_run.sh list
```
