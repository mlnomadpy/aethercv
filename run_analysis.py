import typing as tp
import logging
from pathlib import Path

import tensorflow_datasets as tfds
import tensorflow as tf
import optax
import jax

from utils.config import DATASET_CONFIGS, mesh
from utils.training_utils import model_loop
from utils.logging import setup_logging
from utils.yaml_config import AetherCVConfig, config_to_legacy_format
from utils.analysis_utils import (
    compare_training_curves,
    print_final_metrics_comparison_all,
    analyze_convergence_all,
    detailed_test_evaluation_all,
    plot_confusion_matrices_all,
    generate_summary_report_all,
    visualize_kernels_all,
    activation_map_visualization_all,
    saliency_map_analysis_all,
    create_viz_directory
)
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

# Suppress excessive TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def get_optimizer(optimizer_name: str, learning_rate: float) -> optax.GradientTransformation:
    """Returns an Optax optimizer based on its name."""
    if optimizer_name.lower() == 'adamw':
        return optax.adamw(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'adam':
        return optax.adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return optax.sgd(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: '{optimizer_name}'")

def _get_dataset_info(dataset_name: str) -> tp.Dict[str, tp.Any]:
    """Fetches dataset information like class names and batch size."""
    dataset_config_info = DATASET_CONFIGS.get(dataset_name, {})
    class_names_comp = []
    
    if not dataset_config_info:
        print(f"Warning: Dataset '{dataset_name}' not in DATASET_CONFIGS. Some features might use defaults or fail.")
        try:
            _, ds_info_comp_fallback = tfds.load(dataset_name, split='train', with_info=True, as_supervised=False)
            class_names_comp = ds_info_comp_fallback.features['label'].names
        except Exception as e:
            print(f"Could not load fallback dataset info for {dataset_name}: {e}. Using dummy info.")
            num_classes_fallback = 10 # A reasonable default
            class_names_comp = [f"Class {i}" for i in range(num_classes_fallback)]
    else:
        try:
            train_split_name = dataset_config_info.get('train_split', 'train')
            _, ds_info_comp = tfds.load(dataset_name, split=train_split_name, with_info=True, as_supervised=False)
            label_key_to_use = dataset_config_info.get('label_key', 'label')
            class_names_comp = ds_info_comp.features[label_key_to_use].names
        except Exception as e:
            print(f"Warning: Could not get class names for {dataset_name}: {e}. Using generic names from config.")
            num_classes_from_config = dataset_config_info.get('num_classes', 10)
            class_names_comp = [f"Class {i}" for i in range(num_classes_from_config)]

    default_batch_size = 128
    if mesh:
        default_batch_size = 64 * mesh.devices.shape[0]
        
    batch_size = dataset_config_info.get('batch_size', default_batch_size)

    return {'class_names': class_names_comp, 'batch_size': batch_size}


def _run_combined_analysis(config: AetherCVConfig, trained_model_data: dict, class_names: list, batch_size_for_eval: int):
    """Runs all the analysis functions based on the configuration."""
    print("\n📊 Running analysis...")

    model_configs = config_to_legacy_format(config)
    model_names_list = [cfg['display_name'] for cfg in model_configs]
    histories_list = [trained_model_data[name]['history'] for name in model_names_list]
    models_list = [trained_model_data[name]['model'] for name in model_names_list]
    layer_name_map_for_kernel_viz = {name: trained_model_data[name]['kernel_viz_layer'] for name in model_names_list}
    layer_name_map_for_activation_viz = {name: trained_model_data[name]['activation_viz_layer'] for name in model_names_list}
    
    viz_dir = create_viz_directory(config.experiment.name or f"comparison_{config.dataset.name}")
    analysis_mode = config.analysis.mode

    # Basic Analysis
    compare_training_curves(histories_list, model_names_list, save_path=viz_dir / "training_curves.png")
    print_final_metrics_comparison_all(histories_list, model_names_list)
    analyze_convergence_all(histories_list, model_names_list)

    # Standard Analysis
    if analysis_mode in ['standard', 'advanced', 'comprehensive']:
        detailed_test_evaluation_all(models_list, model_names_list, config.dataset.name, batch_size=batch_size_for_eval, save_path=viz_dir)
        plot_confusion_matrices_all(models_list, model_names_list, config.dataset.name, class_names, save_path=viz_dir)
        visualize_kernels_all(models_list, layer_name_map_for_kernel_viz, save_path=viz_dir)
        activation_map_visualization_all(models_list, layer_name_map_for_activation_viz, config.dataset.name, save_path=viz_dir)
        saliency_map_analysis_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)

    # Advanced Analysis
    if analysis_mode in ['advanced', 'comprehensive']:
        grad_cam_analysis_all(models_list, model_names_list, config.dataset.name, layer_name_map_for_activation_viz, save_path=viz_dir)
        guided_backprop_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        integrated_gradients_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        feature_space_visualization_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        layer_wise_activation_analysis_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        model_decision_boundary_comparison(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        prediction_uncertainty_analysis_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        filter_similarity_analysis_all(models_list, model_names_list, layer_name_map_for_kernel_viz, save_path=viz_dir)
        feature_attribution_analysis_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        information_flow_analysis_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        decision_boundary_visualization_2d(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        prediction_reliability_analysis_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)

    # Comprehensive Analysis
    if analysis_mode == 'comprehensive':
        adversarial_robustness_analysis_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        noise_robustness_analysis_all(models_list, model_names_list, config.dataset.name, save_path=viz_dir)
        out_of_distribution_detection_all(models_list, model_names_list, 'cifar100' if config.dataset.name == 'cifar10' else 'svhn_cropped', save_path=viz_dir)
        computational_efficiency_analysis_all(models_list, model_names_list, config.dataset.name)
        model_complexity_analysis_all(models_list, model_names_list)

    # Final Summary Report
    if config.output.save_metrics:
        generate_summary_report_all(
            model_names=model_names_list,
            histories=histories_list,
            dataset_name=config.dataset.name,
            analysis_mode=analysis_mode,
            save_path=viz_dir / "summary_report.md"
        )
    
    print(f"\n✅ Analysis complete. Visualizations and reports saved in: {viz_dir}")


def run_analysis_pipeline(config: AetherCVConfig) -> tp.Dict[str, tp.Any]:
    """
    Run a complete experiment: training, analysis, and reporting based on a config object.
    """
    # 1. Setup WandB
    if config.wandb.enabled:
        try:
            setup_logging(project_name=config.wandb.project_name, entity=config.wandb.entity)
            print("🔗 WandB logging enabled")
        except Exception as e:
            print(f"⚠️ Warning: Could not setup WandB logging: {e}")
            config.wandb.enabled = False

    # 2. Get model and dataset configurations
    model_configs = config_to_legacy_format(config)
    if not model_configs:
        raise ValueError("Configuration must contain at least one model.")
    
    print(f"\n🚀 Training {len(model_configs)} models on {config.dataset.name.upper()}")
    
    dataset_info = _get_dataset_info(config.dataset.name)
    class_names = dataset_info['class_names']
    batch_size_for_eval = dataset_info['batch_size']

    # 3. Train all models
    trained_model_data = {}
    optimizer_constructor = lambda lr: get_optimizer(config.training.optimizer, lr)

    for model_config in model_configs:
        print(f"\n🚀 Training {model_config['display_name']}...")
        model, history = model_loop(
            model_class_name=model_config['model_class_name'],
            model_name=model_config['display_name'],
            dataset_name=config.dataset.name,
            rng_seed=model_config['rng_seed'],
            learning_rate=config.training.learning_rate,
            optimizer_constructor=optimizer_constructor,
            enable_wandb_logging=config.wandb.enabled,
            experiment_name=config.experiment.name
        )
        trained_model_data[model_config['display_name']] = {
            'model': model,
            'history': history,
            'kernel_viz_layer': model_config['kernel_viz_layer'],
            'activation_viz_layer': model_config['activation_viz_layer']
        }
    
    print(f"\n✅ All models trained on {config.dataset.name}")

    # 4. Run Combined Analysis
    if config.analysis.mode != 'none':
        _run_combined_analysis(
            config=config,
            trained_model_data=trained_model_data,
            class_names=class_names,
            batch_size_for_eval=batch_size_for_eval
        )
    
    return trained_model_data
