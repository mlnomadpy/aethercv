"""AetherCV Models Package.

This package provides a modular architecture for computer vision models
with an abstract base class and specialized implementations.

Structure:
- model.py: Contains the abstract BaseModel class and model factory
- aether/: Advanced models using custom Aether layers
- linear/: Standard models using linear/convolutional layers
"""

from .model import BaseModel, create_model
from .aether import YatCNN, YatResNet
from .linear import LinearCNN, LinearResNet

__all__ = [
    'BaseModel',
    'create_model',
    'YatCNN', 
    'LinearCNN',
    'YatResNet',
    'LinearResNet'
]
