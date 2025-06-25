"""Linear model package.

This package contains models that use standard linear/convolutional
layers for straightforward architectures.
"""

from .linear_cnn import LinearCNN
from .linear_resnet import LinearResNet

__all__ = ['LinearCNN', 'LinearResNet']
