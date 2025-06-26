import typing as tp
from functools import partial
import logging
import argparse
import os
import sys
from pathlib import Path

import tensorflow_datasets as tfds
import tensorflow as tf
import optax
import jax.numpy as jnp
import jax # jax import should be before flax.nnx typically

# Suppress excessive TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import from new modularized components
from utils.config import DATASET_CONFIGS, mesh # mesh is initialized in config.py
from utils.yaml_config import ConfigLoader, AetherCVConfig, config_to_legacy_format
from utils.training_utils import _train_model_loop
from utils.analysis_utils import (
    compare_training_curves,
    print_final_metrics_comparison_all,
    analyze_convergence_all,
    detailed_test_evaluation_all,
    plot_confusion_matrices_all,
    generate_summary_report_all,
    visualize_kernels_all,
    activation_map_visualization_all,
    saliency_map_analysis_all
)
# Import new advanced analysis methods
from utils.advanced_explainability import (
    guided_backprop_all,
    integrated_gradients_all,
    feature_space_visualization_all,
    layer_wise_activation_analysis_all,
    grad_cam_analysis_all,
    model_decision_boundary_comparison,
    prediction_uncertainty_analysis_all
)
from utils.robustness_analysis import (
    adversarial_robustness_analysis_all,
    noise_robustness_analysis_all,
    out_of_distribution_detection_all,
    computational_efficiency_analysis_all,
    model_complexity_analysis_all
)
from utils.interpretability_methods import (
    filter_similarity_analysis_all,
    feature_attribution_analysis_all,
    information_flow_analysis_all,
    decision_boundary_visualization_2d,
    prediction_reliability_analysis_all
)

def run_complete_comparison(dataset_name: str = 'cifar10', model_configs_to_run: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None,
                           analysis_mode: str = 'standard'):
    """
    Run comprehensive model comparison with different levels of analysis.
    
    Args:
        dataset_name: Name of the dataset to use
        model_configs_to_run: List of model configurations to run
        analysis_mode: Level of analysis to perform
            - 'basic': Just training curves and basic metrics
            - 'standard': Basic + activation maps, saliency, confusion matrices  
            - 'advanced': Standard + all explainability methods
            - 'comprehensive': Advanced + robustness and efficiency analysis
    """
    # Determine number of models for display
    num_models = len(model_configs_to_run) if model_configs_to_run else 4
    print("\\n" + "="*80)
    print(f"          COMPREHENSIVE {num_models}-MODEL COMPARISON FOR: {dataset_name.upper()}")
    print("="*80)

    learning_rate = 0.007
    # momentum = 0.9 # Not directly used by adamw in the current setup

    # --- Configuration for all models ---
    default_model_configs = [
        {'model_class_name': "YatCNN",       'display_name': "YatCNN",       'rng_seed': 0, 'kernel_viz_layer': 'conv1',          'activation_viz_layer': 'conv1'},
        {'model_class_name': "LinearCNN",    'display_name': "LinearCNN",    'rng_seed': 0, 'kernel_viz_layer': 'conv1',          'activation_viz_layer': 'conv1'},
        {'model_class_name': "YatResNet",    'display_name': "YatResNet",    'rng_seed': 1, 'kernel_viz_layer': 'stem_conv',      'activation_viz_layer': 'stem_conv'},
        {'model_class_name': "LinearResNet", 'display_name': "LinearResNet", 'rng_seed': 1, 'kernel_viz_layer': 'stem_conv',      'activation_viz_layer': 'stem_relu'},
    ]
    
    # Use provided model configs or default to all models
    model_configs = model_configs_to_run if model_configs_to_run is not None else default_model_configs
    
    if not model_configs:
        raise ValueError("model_configs_to_run cannot be an empty list. Provide at least one model configuration.")

    trained_model_data = {}

    # --- Get Dataset Info (once) ---
    dataset_config_info = DATASET_CONFIGS.get(dataset_name, {})
    if not dataset_config_info:
        print(f"Warning: Dataset '{dataset_name}' not in DATASET_CONFIGS. Some features might use defaults or fail.")
        try:
            _, ds_info_comp_fallback = tfds.load(dataset_name, split='train', with_info=True, as_supervised=False)
        except Exception as e:
            print(f"Could not load fallback dataset info for {dataset_name} using tfds.load: {e}. Using dummy info.")
            class DummyInfo: # Minimal dummy info
                def __init__(self):
                    self.features = {
                        'label': type('DummyLabel', (object,), {'num_classes': 10, 'names': [f'Class {i}' for i in range(10)]})(),
                        'image': type('DummyImage', (object,), {'shape': (None, None, 3)})()
                    }
            ds_info_comp_fallback = DummyInfo()

        try:
            class_names_comp = ds_info_comp_fallback.features['label'].names
        except (KeyError, AttributeError):
            print(f"Could not infer class names for {dataset_name}, using a placeholder list.")
            try:
                num_classes_fallback = ds_info_comp_fallback.features['label'].num_classes
                class_names_comp = [f"Class {i}" for i in range(num_classes_fallback)]
            except: # Last resort fallback
                class_names_comp = [f"Class {i}" for i in range(DATASET_CONFIGS.get('cifar10', {}).get('num_classes', 10))]
    else:
        try:
            # Ensure train_split is correctly fetched or default
            train_split_name = dataset_config_info.get('train_split', 'train')
            _, ds_info_comp = tfds.load(dataset_name, split=train_split_name, with_info=True, as_supervised=False)
        except Exception as e:
            print(f"Could not load primary dataset info for {dataset_name} (split: {train_split_name}) using tfds.load: {e}. Using dummy info based on config.")
            class DummyInfo: # Minimal dummy info based on config
                def __init__(self):
                    num_classes = dataset_config_info.get('num_classes', 10)
                    input_channels = dataset_config_info.get('input_channels', 3)
                    self.features = {
                        'label': type('DummyLabel', (object,), {'num_classes': num_classes, 'names': [f'Class {i}' for i in range(num_classes)]})(),
                        'image': type('DummyImage', (object,), {'shape': (None, None, input_channels)})()
                    }
            ds_info_comp = DummyInfo()

        label_key_to_use = dataset_config_info.get('label_key', 'label')
        try:
            class_names_comp = ds_info_comp.features[label_key_to_use].names
        except (KeyError, AttributeError):
            print(f"Warning: Could not get class names for key '{label_key_to_use}' in {dataset_name}. Using generic names from config.")
            num_classes_from_config = dataset_config_info.get('num_classes', DATASET_CONFIGS.get('cifar10', {}).get('num_classes', 10))
            class_names_comp = [f"Class {i}" for i in range(num_classes_from_config)]

    # Use batch_size from DATASET_CONFIGS (which is already scaled by data parallel dimension)
    # Fallback to a default if dataset_name is not in DATASET_CONFIGS
    default_batch_size = 64 * (mesh.devices.shape[0] if mesh else 1) # mesh might not be available if config import failed
    current_batch_size_for_eval = dataset_config_info.get('batch_size', default_batch_size)


    # --- Train all models ---
    print("\\n" + "="*60)
    print(f"         TRAINING ALL MODELS for {dataset_name.upper()}")
    print("="*60)

    for config_entry in model_configs: # Renamed config to config_entry to avoid conflict with aethercv.utils.config
        print(f"\\n🚀 Training {config_entry['display_name']} Model on {dataset_name}...")
        print("-"*50)
        model, history = _train_model_loop( # _train_model_loop is imported
            model_class_name=config_entry['model_class_name'],
            model_name=config_entry['display_name'],
            dataset_name=dataset_name,
            rng_seed=config_entry['rng_seed'],
            learning_rate=learning_rate,
            # momentum=momentum, # momentum is not directly used by adamw
            optimizer_constructor=lambda lr_opt: optax.adamw(learning_rate=lr_opt) # lr_opt to avoid conflict
        )
        trained_model_data[config_entry['display_name']] = {
            'model': model,
            'history': history,
            'kernel_viz_layer': config_entry['kernel_viz_layer'],
            'activation_viz_layer': config_entry['activation_viz_layer']
        }

    print("\\n" + "="*60)
    print(f"         ALL MODELS TRAINED for {dataset_name.upper()}")
    print("="*60)

    # --- Prepare data for combined analysis functions ---
    model_names_list = [config_entry['display_name'] for config_entry in model_configs]
    histories_list = [trained_model_data[name]['history'] for name in model_names_list]
    models_list = [trained_model_data[name]['model'] for name in model_names_list]
    layer_name_map_for_kernel_viz = {name: trained_model_data[name]['kernel_viz_layer'] for name in model_names_list}
    layer_name_map_for_activation_viz = {name: trained_model_data[name]['activation_viz_layer'] for name in model_names_list}    # --- Run Combined Analysis ---
    print(f"\\n📊 STEP 3: Running Combined Comparison Analysis for ALL models on {dataset_name}...")
    print("-"*70)

    # Create centralized directory for saving all visualizations
    from utils.analysis_utils import create_viz_directory
    global_save_dir = create_viz_directory()
    print(f"📁 All visualizations will be saved to: {global_save_dir}")

    compare_training_curves(histories_list, model_names_list, save_dir=global_save_dir)
    print_final_metrics_comparison_all(histories_list, model_names_list)
    analyze_convergence_all(histories_list, model_names_list)

    # Local preprocess_data_fn for evaluation dataset in run_complete_comparison
    def eval_preprocess_fn_local(sample):
        image = tf.cast(sample['image'], tf.float32) / 255.0
        # No augmentation for eval
        return {'image': image, 'label': sample['label']}

    # Ensure test_split is correctly fetched or default
    test_split_name = dataset_config_info.get('test_split', 'test')
    current_test_ds_for_eval_demo = (
        tfds.load(dataset_name, split=test_split_name, as_supervised=False)
        .map(eval_preprocess_fn_local, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(current_batch_size_for_eval, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    all_models_predictions_package = detailed_test_evaluation_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names=class_names_comp)    # ===== ANALYSIS EXECUTION BASED ON MODE =====
    if analysis_mode in ['basic']:
        print(f"\n📊 BASIC ANALYSIS COMPLETE for {len(model_configs)}-way comparison.")
    elif analysis_mode in ['standard']:
        print(f"\n📊 RUNNING STANDARD ANALYSIS...")
        print("-" * 70)
        
        if all_models_predictions_package:
            plot_confusion_matrices_all(all_models_predictions_package, save_dir=global_save_dir)
            generate_summary_report_all(histories_list, model_names_list, all_models_predictions_package)
        else:
            print("Skipping confusion matrix and summary report generation due to failed detailed test evaluation.")
            generate_summary_report_all(histories_list, model_names_list, {}) # Partial report

        visualize_kernels_all(models_list, model_names_list, layer_name_map_for_kernel_viz, num_kernels_to_show=16, save_dir=global_save_dir)
        activation_map_visualization_all(models_list, model_names_list, layer_name_map_for_activation_viz, current_test_ds_for_eval_demo, num_maps_to_show=16, save_dir=global_save_dir)
        saliency_map_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names=class_names_comp, save_dir=global_save_dir)
        
        print(f"\n📊 STANDARD ANALYSIS COMPLETE for {len(model_configs)}-way comparison.")
    elif analysis_mode in ['advanced']:
        print(f"\n📊 RUNNING STANDARD ANALYSIS...")
        print("-" * 70)
        
        if all_models_predictions_package:
            plot_confusion_matrices_all(all_models_predictions_package, save_dir=global_save_dir)
            generate_summary_report_all(histories_list, model_names_list, all_models_predictions_package)
        else:
            print("Skipping confusion matrix and summary report generation due to failed detailed test evaluation.")
            generate_summary_report_all(histories_list, model_names_list, {}) # Partial report

        visualize_kernels_all(models_list, model_names_list, layer_name_map_for_kernel_viz, num_kernels_to_show=16, save_dir=global_save_dir)
        activation_map_visualization_all(models_list, model_names_list, layer_name_map_for_activation_viz, current_test_ds_for_eval_demo, num_maps_to_show=16, save_dir=global_save_dir)
        saliency_map_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names=class_names_comp, save_dir=global_save_dir)
        
        print(f"\n🚀 RUNNING ADVANCED EXPLAINABILITY ANALYSIS...")
        print("-" * 70)
        
        # Advanced gradient-based methods
        guided_backprop_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        integrated_gradients_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        feature_attribution_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
          # Feature representation analysis
        feature_space_visualization_all(models_list, model_names_list, current_test_ds_for_eval_demo, 
                                       class_names_comp, layer_name_map_for_activation_viz, max_samples=500, save_dir=global_save_dir)
        
        # Model comparison and decision boundaries
        model_decision_boundary_comparison(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        decision_boundary_visualization_2d(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        
        # Uncertainty and reliability analysis
        prediction_uncertainty_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        prediction_reliability_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        
        print(f"\n🚀 ADVANCED ANALYSIS COMPLETE for {len(model_configs)}-way comparison.")
    elif analysis_mode in ['comprehensive']:
        print(f"\n📊 RUNNING STANDARD ANALYSIS...")
        print("-" * 70)
        
        if all_models_predictions_package:
            plot_confusion_matrices_all(all_models_predictions_package, save_dir=global_save_dir)
            generate_summary_report_all(histories_list, model_names_list, all_models_predictions_package)
        else:
            print("Skipping confusion matrix and summary report generation due to failed detailed test evaluation.")
            generate_summary_report_all(histories_list, model_names_list, {}) # Partial report

        visualize_kernels_all(models_list, model_names_list, layer_name_map_for_kernel_viz, num_kernels_to_show=16, save_dir=global_save_dir)
        activation_map_visualization_all(models_list, model_names_list, layer_name_map_for_activation_viz, current_test_ds_for_eval_demo, num_maps_to_show=16, save_dir=global_save_dir)
        saliency_map_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names=class_names_comp, save_dir=global_save_dir)
        
        print(f"\n🚀 RUNNING ADVANCED EXPLAINABILITY ANALYSIS...")
        print("-" * 70)
        
        # Advanced gradient-based methods
        guided_backprop_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        integrated_gradients_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        feature_attribution_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        
        # Feature representation analysis
        feature_space_visualization_all(models_list, model_names_list, current_test_ds_for_eval_demo, 
                                       class_names_comp, layer_name_map_for_activation_viz, max_samples=500, save_dir=global_save_dir)
        
        # Layer-wise analysis
        sample_layers_map = {
            name: [layer_name_map_for_kernel_viz[name], layer_name_map_for_activation_viz[name]] 
            for name in model_names_list
        }
        layer_wise_activation_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, sample_layers_map, save_dir=global_save_dir)
        
        # Information flow analysis
        information_flow_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, sample_layers_map, save_dir=global_save_dir)
        
        # Attention and influence analysis
        grad_cam_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, 
                             class_names_comp, layer_name_map_for_activation_viz, save_dir=global_save_dir)
        
        # Model comparison and decision boundaries
        model_decision_boundary_comparison(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        decision_boundary_visualization_2d(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        
        # Uncertainty and reliability analysis
        prediction_uncertainty_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        prediction_reliability_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        print(f"\n🔬 RUNNING ROBUSTNESS & EFFICIENCY ANALYSIS...")
        print("-" * 70)
        
        # Robustness analysis
        adversarial_robustness_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        noise_robustness_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, class_names_comp, save_dir=global_save_dir)
        
        # Model complexity and efficiency
        model_complexity_analysis_all(models_list, model_names_list, save_dir=global_save_dir)
        computational_efficiency_analysis_all(models_list, model_names_list, current_test_ds_for_eval_demo, save_dir=global_save_dir)
        
        # Filter and architectural analysis
        filter_similarity_analysis_all(models_list, model_names_list, layer_name_map_for_kernel_viz, save_dir=global_save_dir)
        
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE for {len(model_configs)}-way comparison. 🎉")
    else:
        print(f"Unknown analysis_mode: {analysis_mode}. Using 'standard' mode.")
        analysis_mode = 'standard'
        # Recursively call with standard mode
        return run_complete_comparison(dataset_name, model_configs_to_run, 'standard')
    
    print("   Review plots and console output for results.")

    final_results_structure = {
        model_name: {
            # 'model': trained_model_data[model_name]['model'], # Model objects can be large
            'history': trained_model_data[model_name]['history'],
            'kernel_viz_layer': trained_model_data[model_name]['kernel_viz_layer'],
            'activation_viz_layer': trained_model_data[model_name]['activation_viz_layer']
        } for model_name in model_names_list
    }
    return final_results_structure

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
    
    Args:
        model_name: One of 'yat_cnn', 'linear_cnn', 'yat_resnet', 'linear_resnet'
        
    Returns:
        List containing single model configuration
        
    Raises:
        ValueError: If model_name is not recognized
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
    
    return parser.parse_args()

def run_experiment_from_config(config: AetherCVConfig) -> tp.Dict[str, tp.Any]:
    """
    Run an experiment based on YAML configuration.
    
    Args:
        config: AetherCV configuration object
        
    Returns:
        Dictionary containing experiment results
    """
    print("\\n" + "="*80)
    print(f"🧪 EXPERIMENT: {config.experiment.name}")
    print(f"📝 DESCRIPTION: {config.experiment.description}")
    print("="*80)
    
    # Update DATASET_CONFIGS with values from YAML config if needed
    dataset_name = config.dataset.name
    if dataset_name in DATASET_CONFIGS:
        # Override specific values from config
        DATASET_CONFIGS[dataset_name]['num_epochs'] = config.dataset.num_epochs
        DATASET_CONFIGS[dataset_name]['eval_every'] = config.dataset.eval_every
        # Note: batch_size will be handled in run_complete_comparison
    
    # Convert models to legacy format
    model_configs = config_to_legacy_format(config)
    
    print(f"📊 Dataset: {dataset_name}")
    print(f"📈 Models: {[model.display_name for model in config.models]}")
    print(f"🔍 Analysis Mode: {config.analysis.mode}")
    print(f"🎯 Learning Rate: {config.training.learning_rate}")
    print(f"⚙️  Optimizer: {config.training.optimizer}")
    print(f"🔄 Epochs: {config.dataset.num_epochs}")
    print(f"📏 Batch Size: {config.dataset.batch_size}")
    
    # Create output directory if specified
    if config.output.save_dir:
        output_dir = Path(config.output.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output Directory: {output_dir.absolute()}")
    
    # Run the experiment
    results = run_complete_comparison(
        dataset_name=dataset_name,
        model_configs_to_run=model_configs,
        analysis_mode=config.analysis.mode
    )
    
    return results

# ===== QUICK START FUNCTIONS =====
def save_metrics_example():
    print("\\\\n💾 HOW TO SAVE METRICS DURING TRAINING:"); print("-"*50)
    print('''
# After training your YAT model: yat_metrics_history = metrics_history.copy()
# After training your Linear model:  linear_metrics_history = metrics_history.copy()
# Or save to files: import pickle
# with open(\'yat_metrics.pkl\', \'wb\') as f: pickle.dump(yat_metrics_history, f)
# with open(\'linear_metrics.pkl\', \'wb\') as f: pickle.dump(linear_metrics_history, f)
# Load later:
# with open(\'yat_metrics.pkl\', \'rb\') as f: yat_metrics_history = pickle.load(f)
# with open(\'linear_metrics.pkl\', \'rb\') as f: linear_metrics_history = pickle.load(f)
''')

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize config loader
    config_loader = ConfigLoader()
    
    # Handle --list-configs option
    if args.list_configs:
        available_configs = config_loader.list_available_configs()
        print("\\n📋 Available Configuration Files:")
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
        print("\\n💡 Usage: python train.py --config <config_name>")
        sys.exit(0)
    
    # Check if using YAML configuration
    if args.config:
        print("\\n" + "="*80)
        print(f"🔧 USING YAML CONFIGURATION: {args.config}")
        print("="*80)
        
        try:
            config = config_loader.load_config(args.config)
            config_loader.validate_config(config)
            
            # Run experiment from config
            results = run_experiment_from_config(config)
            
            print("\\n🎉 YAML-configured experiment completed successfully!")
            print("\\n📊 Results Summary:")
            if results:
                for model_name, data in results.items():
                    print(f"  ✅ {model_name}: Training completed. History keys: {list(data['history'].keys())}")
            else:
                print("  ❌ No results returned from experiment.")
                
        except Exception as e:
            print(f"\\n❌ Error running experiment with config '{args.config}': {str(e)}")
            print("\\n💡 Use --list-configs to see available configurations")
            sys.exit(1)
    
    else:
        # Use legacy command-line arguments
        print("\\n" + "="*80 + "\\n" + "="*80)
        print(f"🚀 AETHERCV TRAINING PIPELINE (Legacy Mode)")
        print(f"📊 Dataset: {args.dataset}")
        print(f"🔍 Analysis Mode: {args.analysis_mode}")
        
        # Determine which models to run
        if args.model:
            # Single model specified
            print(f"🎯 Training single model: {args.model.upper()}")
            model_configs = get_single_model_config(args.model)
            experiment_type = f"Single Model ({args.model.upper()})"
        else:
            # Model group specified
            print(f"🎯 Training model group: {args.model_group.upper()}")
            model_group_map = {
                'all': get_default_model_configs,
                'cnn': get_cnn_models_only,
                'resnet': get_resnet_models_only,
                'yat': get_yat_models_only,
                'linear': get_linear_models_only,
            }
            model_configs = model_group_map[args.model_group]()
            experiment_type = f"Model Group ({args.model_group.upper()})"
        
        print(f"🔧 Experiment Type: {experiment_type}")
        print(f"📈 Models to train: {[config['display_name'] for config in model_configs]}")
        print("="*80)
        
        # Run the experiment
        print(f"\\n� Starting {experiment_type} experiment on {args.dataset} with {args.analysis_mode} analysis...")
        results = run_complete_comparison(
            dataset_name=args.dataset, 
            model_configs_to_run=model_configs, 
            analysis_mode=args.analysis_mode
        )

        print("\\n🎉 Legacy experiment completed successfully!")
        print("\\n📊 Results Summary:")
        if results:
            for model_name_res, data_res in results.items():
                print(f"  ✅ {model_name_res}: Training completed. History keys: {list(data_res['history'].keys())}")
        else:
            print("  ❌ No results returned from run_complete_comparison.")
    
    # Display usage examples
    print("\\n" + "="*80)
    print("� USAGE EXAMPLES:")
    print("="*80)
    print("\\n🔧 YAML Configuration Mode (Recommended):")
    print("   python train.py --config single_yat_cnn")
    print("   python train.py --config full_comparison")  
    print("   python train.py --config cnn_stl10")
    print("   python train.py --config resnet_comparison")
    print("   python train.py --config eurosat_analysis")
    print("\\n   # List all available configs:")
    print("   python train.py --list-configs")
    
    print("\\n� Legacy Command-Line Mode:")
    print("   # Train single model:")
    print("   python train.py --model yat_cnn --dataset cifar10 --analysis-mode basic")
    print("   python train.py --model linear_resnet --dataset stl10 --analysis-mode standard")
    print("\\n   # Train model groups:")
    print("   python train.py --model-group cnn --dataset cifar10 --analysis-mode advanced")
    print("   python train.py --model-group all --dataset eurosat/rgb --analysis-mode comprehensive")
    
    print("\\n🔧 PROGRAMMATIC API:")
    print("   # Load and run from config:")
    print("   from utils.yaml_config import ConfigLoader")
    print("   loader = ConfigLoader()")
    print("   config = loader.load_config('single_yat_cnn')")
    print("   results = run_experiment_from_config(config)")
    print("\\n   # Legacy API:")
    print("   results = run_complete_comparison(dataset_name='cifar10', model_configs_to_run=get_yat_cnn_only())")
    
    print("\\n📁 Check the generated visualization directory for detailed analysis plots and reports.")
    print("="*80)
