"""Aether model package.

This package contains models that use custom Aether layers
and advanced architectures for enhanced expressiveness.
"""

from .yat_cnn import YatCNN
from .yat_resnet import YatResNet

__all__ = ['YatCNN', 'YatResNet']
