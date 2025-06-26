"""
Logging configuration for AetherCV
"""

# WandB Configuration
WANDB_CONFIG = {
    "project_name": "aethercv",
    "entity": None,  # Set to your WandB username or team name
}

# Logging settings
LOGGING_CONFIG = {
    "log_frequency": 10,  # Log every N steps
    "save_model_frequency": 100,  # Save model every N steps
    "log_images": True,
    "log_gradients": True,
    "log_parameters": True,
}

# Model-specific configurations
MODEL_CONFIGS = {
    "yat_cnn": {
        "tags": ["yat", "cnn", "image-classification", "aether"],
        "notes": "YAT CNN model for image classification with Aether layer"
    },
    "yat_resnet": {
        "tags": ["yat", "resnet", "image-classification", "aether"],
        "notes": "YAT ResNet model for image classification with Aether layer"
    },
    "linear_cnn": {
        "tags": ["linear", "cnn", "image-classification", "baseline"],
        "notes": "Linear CNN model for baseline comparison"
    },
    "linear_resnet": {
        "tags": ["linear", "resnet", "image-classification", "baseline"],
        "notes": "Linear ResNet model for baseline comparison"
    }
}

# Dataset-specific configurations
DATASET_CONFIGS = {
    "stl10": {
        "dataset_name": "STL-10",
        "num_classes": 10,
        "image_size": 96
    },
    "eurosat": {
        "dataset_name": "EuroSAT",
        "num_classes": 10,
        "image_size": 64
    },
    "cifar10": {
        "dataset_name": "CIFAR-10",
        "num_classes": 10,
        "image_size": 32
    }
}
