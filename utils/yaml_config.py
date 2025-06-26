"""
YAML Configuration Management for AetherCV
Handles loading and validation of YAML configuration files.
"""

import yaml
import os
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for experiment metadata."""
    name: str = "default_experiment"
    description: str = ""


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""
    name: str = "cifar10"
    num_epochs: int = 5
    eval_every: int = 200
    batch_size: int = 64


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_class_name: str
    display_name: str
    rng_seed: int = 0
    kernel_viz_layer: str = "conv1"
    activation_viz_layer: str = "conv1"


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 0.007
    optimizer: str = "adamw"


@dataclass
class AnalysisConfig:
    """Configuration for analysis and evaluation."""
    mode: str = "standard"
    save_visualizations: bool = True


@dataclass
class OutputConfig:
    """Configuration for output and saving."""
    save_dir: tp.Optional[str] = None
    save_models: bool = True
    save_metrics: bool = True


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases logging."""
    enabled: bool = True
    project_name: str = "aethercv"
    entity: tp.Optional[str] = None
    tags: tp.List[str] = field(default_factory=list)
    notes: str = ""
    log_frequency: int = 10
    log_gradients: bool = True
    log_parameters: bool = True
    log_images: bool = True


@dataclass
class AetherCVConfig:
    """Complete configuration for AetherCV experiments."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    models: tp.List[ModelConfig] = field(default_factory=list)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


class ConfigLoader:
    """Loads and validates YAML configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        
    def list_available_configs(self) -> tp.List[str]:
        """List all available configuration files."""
        if not self.config_dir.exists():
            return []
        
        configs = []
        for file_path in self.config_dir.glob("*.yaml"):
            configs.append(file_path.stem)
        for file_path in self.config_dir.glob("*.yml"):
            configs.append(file_path.stem)
        
        return sorted(configs)
    
    def load_config(self, config_name: str) -> AetherCVConfig:
        """
        Load a configuration from a YAML file.
        
        Args:
            config_name: Name of the config file (without extension)
            
        Returns:
            AetherCVConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If config validation fails
        """
        # Try both .yaml and .yml extensions
        config_path = None
        for ext in ['.yaml', '.yml']:
            potential_path = self.config_dir / f"{config_name}{ext}"
            if potential_path.exists():
                config_path = potential_path
                break
        
        if config_path is None:
            available = self.list_available_configs()
            raise FileNotFoundError(
                f"Configuration '{config_name}' not found in {self.config_dir}. "
                f"Available configs: {available}"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        return self._parse_config(raw_config, config_name)
    
    def _parse_config(self, raw_config: dict, config_name: str) -> AetherCVConfig:
        """Parse raw YAML data into structured config objects."""
        try:
            # Parse experiment config
            exp_data = raw_config.get('experiment', {})
            experiment = ExperimentConfig(
                name=exp_data.get('name', config_name),
                description=exp_data.get('description', '')
            )
            
            # Parse dataset config
            ds_data = raw_config.get('dataset', {})
            dataset = DatasetConfig(
                name=ds_data.get('name', 'cifar10'),
                num_epochs=ds_data.get('num_epochs', 5),
                eval_every=ds_data.get('eval_every', 200),
                batch_size=ds_data.get('batch_size', 64)
            )
            
            # Parse models config
            models_data = raw_config.get('models', [])
            if not models_data:
                raise ValueError("No models specified in configuration")
            
            models = []
            for model_data in models_data:
                if 'model_class_name' not in model_data:
                    raise ValueError("model_class_name is required for each model")
                
                model = ModelConfig(
                    model_class_name=model_data['model_class_name'],
                    display_name=model_data.get('display_name', model_data['model_class_name']),
                    rng_seed=model_data.get('rng_seed', 0),
                    kernel_viz_layer=model_data.get('kernel_viz_layer', 'conv1'),
                    activation_viz_layer=model_data.get('activation_viz_layer', 'conv1')
                )
                models.append(model)
            
            # Parse training config
            train_data = raw_config.get('training', {})
            training = TrainingConfig(
                learning_rate=train_data.get('learning_rate', 0.007),
                optimizer=train_data.get('optimizer', 'adamw')
            )
            
            # Parse analysis config
            analysis_data = raw_config.get('analysis', {})
            analysis = AnalysisConfig(
                mode=analysis_data.get('mode', 'standard'),
                save_visualizations=analysis_data.get('save_visualizations', True)
            )
            
            # Parse output config
            output_data = raw_config.get('output', {})
            output = OutputConfig(
                save_dir=output_data.get('save_dir'),
                save_models=output_data.get('save_models', True),
                save_metrics=output_data.get('save_metrics', True)
            )
            
            # Parse WandB config
            wandb_data = raw_config.get('wandb', {})
            wandb = WandBConfig(
                enabled=wandb_data.get('enabled', True),
                project_name=wandb_data.get('project_name', 'aethercv'),
                entity=wandb_data.get('entity'),
                tags=wandb_data.get('tags', []),
                notes=wandb_data.get('notes', ''),
                log_frequency=wandb_data.get('log_frequency', 10),
                log_gradients=wandb_data.get('log_gradients', True),
                log_parameters=wandb_data.get('log_parameters', True),
                log_images=wandb_data.get('log_images', True)
            )
            
            return AetherCVConfig(
                experiment=experiment,
                dataset=dataset,
                models=models,
                training=training,
                analysis=analysis,
                output=output,
                wandb=wandb
            )
            
        except Exception as e:
            raise ValueError(f"Error parsing configuration '{config_name}': {str(e)}")
    
    def validate_config(self, config: AetherCVConfig) -> None:
        """
        Validate a configuration object.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Validate analysis mode
        valid_modes = ['basic', 'standard', 'advanced', 'comprehensive']
        if config.analysis.mode not in valid_modes:
            raise ValueError(f"Invalid analysis mode '{config.analysis.mode}'. "
                           f"Valid modes: {valid_modes}")
        
        # Validate optimizer
        valid_optimizers = ['adamw', 'adam', 'sgd']
        if config.training.optimizer not in valid_optimizers:
            raise ValueError(f"Invalid optimizer '{config.training.optimizer}'. "
                           f"Valid optimizers: {valid_optimizers}")
        
        # Validate model class names
        valid_models = ['YatCNN', 'LinearCNN', 'YatResNet', 'LinearResNet']
        for model in config.models:
            if model.model_class_name not in valid_models:
                raise ValueError(f"Invalid model class '{model.model_class_name}'. "
                               f"Valid models: {valid_models}")
        
        # Validate learning rate
        if config.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate epochs and batch size
        if config.dataset.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if config.dataset.batch_size <= 0:
            raise ValueError("Batch size must be positive")


def create_default_config() -> AetherCVConfig:
    """Create a default configuration."""
    return AetherCVConfig(
        experiment=ExperimentConfig(
            name="default_experiment",
            description="Default AetherCV experiment"
        ),
        dataset=DatasetConfig(
            name="cifar10",
            num_epochs=5,
            eval_every=200,
            batch_size=64
        ),
        models=[
            ModelConfig(
                model_class_name="YatCNN",
                display_name="YatCNN",
                rng_seed=0,
                kernel_viz_layer="conv1",
                activation_viz_layer="conv1"
            )
        ],
        training=TrainingConfig(
            learning_rate=0.007,
            optimizer="adamw"
        ),
        analysis=AnalysisConfig(
            mode="standard",
            save_visualizations=True
        ),
        output=OutputConfig(
            save_dir=None,
            save_models=True,
            save_metrics=True
        ),
        wandb=WandBConfig(
            enabled=True,
            project_name="aethercv",
            entity=None,
            tags=[],
            notes="",
            log_frequency=10,
            log_gradients=True,
            log_parameters=True,
            log_images=True
        )
    )


def config_to_legacy_format(config: AetherCVConfig) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Convert new config format to legacy model_configs format.
    
    Args:
        config: AetherCV configuration
        
    Returns:
        List of model configurations in legacy format
    """
    legacy_configs = []
    for model in config.models:
        legacy_config = {
            'model_class_name': model.model_class_name,
            'display_name': model.display_name,
            'rng_seed': model.rng_seed,
            'kernel_viz_layer': model.kernel_viz_layer,
            'activation_viz_layer': model.activation_viz_layer
        }
        legacy_configs.append(legacy_config)
    
    return legacy_configs
