import typing as tp
import argparse
import sys
import logging

from utils.yaml_config import (
    ConfigLoader, AetherCVConfig, ExperimentConfig, DatasetConfig, 
    ModelConfig, TrainingConfig, AnalysisConfig, OutputConfig, WandBConfig
)
from run_analysis import run_analysis_pipeline

# Suppress excessive TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ===== HELPER FUNCTIONS FOR MODEL CONFIGURATIONS =====
def get_default_model_configs() -> tp.List[tp.Dict[str, tp.Any]]:
    """Returns the default configuration for all 4 models."""
    return [
        {'model_class_name': "YatCNN",       'display_name': "YatCNN",       'rng_seed': 0, 'kernel_viz_layer': 'conv1',          'activation_viz_layer': 'conv1'},
        {'model_class_name': "LinearCNN",    'display_name': "LinearCNN",    'rng_seed': 0, 'kernel_viz_layer': 'conv1',          'activation_viz_layer': 'conv1'},
        {'model_class_name': "YatResNet",    'display_name': "YatResNet",    'rng_seed': 1, 'kernel_viz_layer': 'stem_conv',      'activation_viz_layer': 'stem_conv'},
        {'model_class_name': "LinearResNet", 'display_name': "LinearResNet", 'rng_seed': 1, 'kernel_viz_layer': 'stem_conv',      'activation_viz_layer': 'stem_relu'},
    ]
    
def get_cnn_models_only() -> tp.List[tp.Dict[str, tp.Any]]:
    """Returns configuration for CNN models only (YatCNN and LinearCNN)."""
    return [
        {'model_class_name': "YatCNN",    'display_name': "YatCNN",    'rng_seed': 0, 'kernel_viz_layer': 'conv1', 'activation_viz_layer': 'conv1'},
        {'model_class_name': "LinearCNN", 'display_name': "LinearCNN", 'rng_seed': 0, 'kernel_viz_layer': 'conv1', 'activation_viz_layer': 'conv1'},
    ]

def get_resnet_models_only() -> tp.List[tp.Dict[str, tp.Any]]:
    """Returns configuration for ResNet models only (YatResNet and LinearResNet)."""
    return [
        {'model_class_name': "YatResNet",    'display_name': "YatResNet",    'rng_seed': 1, 'kernel_viz_layer': 'stem_conv', 'activation_viz_layer': 'stem_conv'},
        {'model_class_name': "LinearResNet", 'display_name': "LinearResNet", 'rng_seed': 1, 'kernel_viz_layer': 'stem_conv', 'activation_viz_layer': 'stem_relu'},
    ]

def get_yat_models_only() -> tp.List[tp.Dict[str, tp.Any]]:
    """Returns configuration for YAT models only (YatCNN and YatResNet)."""
    return [
        {'model_class_name': "YatCNN",    'display_name': "YatCNN",    'rng_seed': 0, 'kernel_viz_layer': 'conv1',     'activation_viz_layer': 'conv1'},
        {'model_class_name': "YatResNet", 'display_name': "YatResNet", 'rng_seed': 1, 'kernel_viz_layer': 'stem_conv', 'activation_viz_layer': 'stem_conv'},
    ]

def get_linear_models_only() -> tp.List[tp.Dict[str, tp.Any]]:
    """Returns configuration for Linear models only (LinearCNN and LinearResNet)."""
    return [
        {'model_class_name': "LinearCNN",    'display_name': "LinearCNN",    'rng_seed': 0, 'kernel_viz_layer': 'conv1',     'activation_viz_layer': 'conv1'},
        {'model_class_name': "LinearResNet", 'display_name': "LinearResNet", 'rng_seed': 1, 'kernel_viz_layer': 'stem_conv', 'activation_viz_layer': 'stem_relu'},
    ]

# ===== INDIVIDUAL MODEL CONFIGURATIONS =====
def get_yat_cnn_only() -> tp.List[tp.Dict[str, tp.Any]]:
    """Returns configuration for YatCNN model only."""
    return [
        {'model_class_name': "YatCNN", 'display_name': "YatCNN", 'rng_seed': 0, 'kernel_viz_layer': 'conv1', 'activation_viz_layer': 'conv1'}
    ]

def get_linear_cnn_only() -> tp.List[tp.Dict[str, tp.Any]]:
    """Returns configuration for LinearCNN model only."""
    return [
        {'model_class_name': "LinearCNN", 'display_name': "LinearCNN", 'rng_seed': 0, 'kernel_viz_layer': 'conv1', 'activation_viz_layer': 'conv1'}
    ]

def get_yat_resnet_only() -> tp.List[tp.Dict[str, tp.Any]]:
    """Returns configuration for YatResNet model only."""
    return [
        {'model_class_name': "YatResNet", 'display_name': "YatResNet", 'rng_seed': 1, 'kernel_viz_layer': 'stem_conv', 'activation_viz_layer': 'stem_conv'}
    ]

def get_linear_resnet_only() -> tp.List[tp.Dict[str, tp.Any]]:
    """Returns configuration for LinearResNet model only."""
    return [
        {'model_class_name': "LinearResNet", 'display_name': "LinearResNet", 'rng_seed': 1, 'kernel_viz_layer': 'stem_conv', 'activation_viz_layer': 'stem_relu'}
    ]

def get_single_model_config(model_name: str) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Get configuration for a single model by name.
    """
    model_map = {
        'yat_cnn': get_yat_cnn_only,
        'linear_cnn': get_linear_cnn_only,
        'yat_resnet': get_yat_resnet_only,
        'linear_resnet': get_linear_resnet_only,
    }
    
    if model_name.lower() not in model_map:
        available_models = ', '.join(model_map.keys())
        raise ValueError(f"Unknown model name '{model_name}'. Available models: {available_models}")
    
    return model_map[model_name.lower()]()

def _create_config_from_args(args: argparse.Namespace) -> AetherCVConfig:
    """Creates an AetherCVConfig object from legacy command-line arguments."""
    
    # Determine which models to run
    if args.model:
        model_configs_legacy = get_single_model_config(args.model)
        exp_name = f"Single Model ({args.model.upper()}) on {args.dataset}"
    else:
        model_group_map = {
            'all': get_default_model_configs,
            'cnn': get_cnn_models_only,
            'resnet': get_resnet_models_only,
            'yat': get_yat_models_only,
            'linear': get_linear_models_only,
        }
        model_configs_legacy = model_group_map[args.model_group]()
        exp_name = f"Model Group ({args.model_group.upper()}) on {args.dataset}"

    # Convert legacy model configs to new ModelConfig objects
    models = [ModelConfig(**mc) for mc in model_configs_legacy]

    return AetherCVConfig(
        experiment=ExperimentConfig(name=exp_name),
        dataset=DatasetConfig(name=args.dataset),
        models=models,
        training=TrainingConfig(), # Uses defaults
        analysis=AnalysisConfig(mode=args.analysis_mode),
        output=OutputConfig(), # Uses defaults
        wandb=WandBConfig(enabled=not args.no_wandb)
    )

# ===== COMMAND LINE ARGUMENT PARSING =====
def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train and compare neural network models')
    
    # Configuration file option
    parser.add_argument('--config', type=str, 
                       help='YAML configuration file to use (from configs/ directory, without .yaml extension)')
    
    parser.add_argument('--list-configs', action='store_true',
                       help='List all available configuration files and exit')
    
    # Legacy command-line options (used when --config is not specified)
    parser.add_argument('--dataset', type=str, default='stl10',
                       help='Dataset to use for training (default: stl10)')
    
    parser.add_argument('--model', type=str, choices=['yat_cnn', 'linear_cnn', 'yat_resnet', 'linear_resnet'], 
                       help='Train only a specific model instead of comparing multiple models')
    
    parser.add_argument('--model-group', type=str, 
                       choices=['all', 'cnn', 'resnet', 'yat', 'linear'],
                       default='cnn',
                       help='Group of models to train (default: cnn). Ignored if --model is specified.')
    
    parser.add_argument('--analysis-mode', type=str, 
                       choices=['basic', 'standard', 'advanced', 'comprehensive'],
                       default='standard',
                       help='Level of analysis to perform (default: standard)')
    
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable WandB logging for this run')
    
    return parser.parse_args()

def main():
    """Main function to parse arguments and run experiments."""
    args = parse_arguments()
    config_loader = ConfigLoader()

    if args.list_configs:
        available_configs = config_loader.list_available_configs()
        print("\n📋 Available Configuration Files:")
        print("="*50)
        if available_configs:
            for config_name in available_configs:
                try:
                    config = config_loader.load_config(config_name)
                    print(f"  📄 {config_name}")
                    print(f"     ├─ Name: {config.experiment.name}")
                    print(f"     ├─ Dataset: {config.dataset.name}")
                    print(f"     ├─ Models: {[m.display_name for m in config.models]}")
                    print(f"     └─ Analysis: {config.analysis.mode}")
                except Exception as e:
                    print(f"  ❌ {config_name} (Error: {str(e)})")
        else:
            print("  No configuration files found in configs/ directory")
        print("\n💡 Usage: python train.py --config <config_name>")
        sys.exit(0)

    config: AetherCVConfig

    if args.config:
        print(f"\n🔧 USING YAML CONFIGURATION: {args.config}")
        try:
            config = config_loader.load_config(args.config)
            config_loader.validate_config(config)
            if args.no_wandb:
                config.wandb.enabled = False
                print("🚫 WandB logging disabled by --no-wandb flag")
        except Exception as e:
            print(f"\n❌ Error with config file: {e}")
            sys.exit(1)
    else:
        print("\n🚀 USING LEGACY COMMAND-LINE ARGUMENTS")
        config = _create_config_from_args(args)

    print("\n" + "="*80)
    print(f"🧪 EXPERIMENT: {config.experiment.name}")
    if config.experiment.description:
        print(f"📝 DESCRIPTION: {config.experiment.description}")
    print(f"📊 Dataset: {config.dataset.name}")
    print(f"📈 Models: {[model.display_name for model in config.models]}")
    print(f"🔍 Analysis Mode: {config.analysis.mode}")
    print("="*80)

    try:
        results = run_analysis_pipeline(config)
        print("\n🎉 Experiment completed successfully!")
        if results:
            print("\n📊 Results Summary:")
            for model_name, data in results.items():
                print(f"  ✅ {model_name}: Training completed. History keys: {list(data['history'].keys())}")
        else:
            print("  ℹ️ No results returned from experiment pipeline.")
    except Exception as e:
        print(f"\n❌ An error occurred during the experiment: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()