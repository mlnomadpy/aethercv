"""
Weights & Biases logging system for AetherCV models
"""

import wandb
import os
from typing import Dict, Any, Optional
from datetime import datetime

class WandBLogger:
    """
    Weights & Biases logger for AetherCV models
    """
    
    def __init__(self, project_name: str = "aethercv", entity: Optional[str] = None):
        """
        Initialize WandB logger
        
        Args:
            project_name: Name of the WandB project
            entity: WandB entity (username or team name)
        """
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.is_initialized = False
        
    def init_run(self, 
                 model_name: str,
                 config: Dict[str, Any] = None,
                 tags: list = None,
                 notes: str = None):
        """
        Initialize a new WandB run for a specific model
        
        Args:
            model_name: Name of the model being trained
            config: Configuration dictionary to log
            tags: List of tags for the run
            notes: Notes for the run
        """
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config=config,
            tags=tags or [model_name],
            notes=notes,
            reinit=True
        )
        self.is_initialized = True
        print(f"WandB run initialized: {run_name}")
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to WandB
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.is_initialized:
            print("Warning: WandB not initialized. Call init_run() first.")
            return
            
        wandb.log(metrics, step=step)
        
    def log_model_architecture(self, model, model_name: str):
        """
        Log model architecture to WandB
        
        Args:
            model: The model to log
            model_name: Name of the model
        """
        if not self.is_initialized:
            print("Warning: WandB not initialized. Call init_run() first.")
            return
            
        try:
            # Log model summary
            wandb.watch(model, log="all", log_freq=100)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.log_metrics({
                f"{model_name}/total_parameters": total_params,
                f"{model_name}/trainable_parameters": trainable_params
            })
            
        except Exception as e:
            print(f"Error logging model architecture: {e}")
            
    def log_images(self, images: Dict[str, Any], step: Optional[int] = None):
        """
        Log images to WandB
        
        Args:
            images: Dictionary of images to log (name: image)
            step: Optional step number
        """
        if not self.is_initialized:
            print("Warning: WandB not initialized. Call init_run() first.")
            return
            
        wandb_images = {}
        for name, img in images.items():
            wandb_images[name] = wandb.Image(img)
            
        wandb.log(wandb_images, step=step)
        
    def log_artifact(self, artifact_path: str, artifact_name: str, artifact_type: str = "model"):
        """
        Log an artifact to WandB
        
        Args:
            artifact_path: Path to the artifact file
            artifact_name: Name of the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
        """
        if not self.is_initialized:
            print("Warning: WandB not initialized. Call init_run() first.")
            return
            
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)
        
    def finish_run(self):
        """
        Finish the current WandB run
        """
        if self.is_initialized:
            wandb.finish()
            self.is_initialized = False
            print("WandB run finished")


class ModelLogger:
    """
    Model-specific logger that wraps WandBLogger
    """
    
    def __init__(self, model_name: str, wandb_logger: WandBLogger):
        """
        Initialize model logger
        
        Args:
            model_name: Name of the model
            wandb_logger: WandBLogger instance
        """
        self.model_name = model_name
        self.wandb_logger = wandb_logger
        self.step = 0
        
    def log_training_metrics(self, loss: float, accuracy: float = None, lr: float = None):
        """
        Log training metrics
        
        Args:
            loss: Training loss
            accuracy: Training accuracy (optional)
            lr: Learning rate (optional)
        """
        metrics = {f"{self.model_name}/train_loss": loss}
        
        if accuracy is not None:
            metrics[f"{self.model_name}/train_accuracy"] = accuracy
        if lr is not None:
            metrics[f"{self.model_name}/learning_rate"] = lr
            
        self.wandb_logger.log_metrics(metrics, step=self.step)
        
    def log_validation_metrics(self, val_loss: float, val_accuracy: float = None):
        """
        Log validation metrics
        
        Args:
            val_loss: Validation loss
            val_accuracy: Validation accuracy (optional)
        """
        metrics = {f"{self.model_name}/val_loss": val_loss}
        
        if val_accuracy is not None:
            metrics[f"{self.model_name}/val_accuracy"] = val_accuracy
            
        self.wandb_logger.log_metrics(metrics, step=self.step)
        
    def log_custom_metrics(self, metrics: Dict[str, Any]):
        """
        Log custom metrics with model name prefix
        
        Args:
            metrics: Dictionary of metrics to log
        """
        prefixed_metrics = {f"{self.model_name}/{k}": v for k, v in metrics.items()}
        self.wandb_logger.log_metrics(prefixed_metrics, step=self.step)
        
    def increment_step(self):
        """
        Increment the step counter
        """
        self.step += 1
        
    def set_step(self, step: int):
        """
        Set the step counter
        
        Args:
            step: Step number to set
        """
        self.step = step


# Global logger instance
_global_wandb_logger = None

def get_global_logger() -> WandBLogger:
    """
    Get or create the global WandB logger instance
    
    Returns:
        WandBLogger instance
    """
    global _global_wandb_logger
    if _global_wandb_logger is None:
        _global_wandb_logger = WandBLogger()
    return _global_wandb_logger

def create_model_logger(model_name: str, config: Dict[str, Any] = None) -> ModelLogger:
    """
    Create a model-specific logger
    
    Args:
        model_name: Name of the model
        config: Configuration to log for this model
        
    Returns:
        ModelLogger instance
    """
    wandb_logger = get_global_logger()
    
    # Initialize run if not already done
    if not wandb_logger.is_initialized:
        wandb_logger.init_run(model_name, config=config)
    
    return ModelLogger(model_name, wandb_logger)

def setup_logging(project_name: str = "aethercv", entity: Optional[str] = None):
    """
    Setup global logging configuration
    
    Args:
        project_name: WandB project name
        entity: WandB entity (username or team)
    """
    global _global_wandb_logger
    _global_wandb_logger = WandBLogger(project_name, entity)
    
    # Set WandB environment variables if not set
    if not os.getenv("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not found in environment variables.")
        print("Please set it by running: wandb login")
