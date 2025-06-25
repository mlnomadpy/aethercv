import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

# Imports from this project
from utils.config import mesh, P, NamedSharding, default_kernel_init, default_bias_init, default_alpha_init
from models.model import BaseModel

# Custom layer imports
from nmn.nnx.nmn import YatNMN
from nmn.nnx.yatconv import YatConv


class YatCNN(BaseModel):
    """YAT (Yet Another Transformer) based CNN model.
    
    This model uses YatConv layers which implement custom convolution
    operations with alpha parameters for enhanced expressiveness.
    """
    
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        super().__init__(num_classes=num_classes, input_channels=input_channels, rngs=rngs)
        
        # Kernel (KH, KW, Cin, Cout) -> P(None, None, None, 'model')
        # Bias (Cout,) -> P('model')
        # Alpha (1,) -> P(None, 'model')
        conv_kwargs = {
            'kernel_init': nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, None, None, 'model'))),
            'bias_init': nnx.with_partitioning(default_bias_init, NamedSharding(mesh, P('model'))),
            'alpha_init': nnx.with_partitioning(default_alpha_init, NamedSharding(mesh, P(None,'model')))
        }
        
        self.conv1 = YatConv(input_channels, 32, kernel_size=(5, 5), rngs=rngs, **conv_kwargs)
        self.conv2 = YatConv(32, 64, kernel_size=(5, 5), rngs=rngs, **conv_kwargs)
        self.conv3 = YatConv(64, 128, kernel_size=(5, 5), rngs=rngs, **conv_kwargs)
        self.conv4 = YatConv(128, 256, kernel_size=(5, 5), rngs=rngs, **conv_kwargs)
        self.conv5 = YatConv(256, num_classes, kernel_size=(5, 5), rngs=rngs, **conv_kwargs)
        
        self.dropout1 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout4 = nnx.Dropout(rate=0.3, rngs=rngs)
        
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None):
        """Forward pass through the YAT CNN model.
        
        Args:
            x: Input tensor of shape (batch, height, width, channels)
            training: Whether the model is in training mode
            return_activations_for_layer: Optional layer name to return activations for
            
        Returns:
            Model output or activations for specified layer
        """
        activations = {}
        
        x = self.conv1(x)
        activations['conv1'] = x
        if return_activations_for_layer == 'conv1': 
            return x
        x = self.dropout1(x, deterministic=not training)
        x = self.avg_pool(x)

        x = self.conv2(x)
        activations['conv2'] = x
        if return_activations_for_layer == 'conv2': 
            return x
        x = self.dropout2(x, deterministic=not training)
        x = self.avg_pool(x)

        x = self.conv3(x)
        activations['conv3'] = x
        if return_activations_for_layer == 'conv3': 
            return x
        x = self.dropout3(x, deterministic=not training)
        x = self.avg_pool(x)

        x = self.conv4(x)
        activations['conv4'] = x
        if return_activations_for_layer == 'conv4': 
            return x
        x = self.dropout3(x, deterministic=not training)  # Note: Original code uses dropout3 here
        x = self.avg_pool(x)

        x = self.conv5(x)
        activations['conv5'] = x
        if return_activations_for_layer == 'conv5': 
            return x

        x = jnp.mean(x, axis=(1, 2))
        activations['global_avg_pool'] = x
        if return_activations_for_layer == 'global_avg_pool': 
            return x

        if return_activations_for_layer is not None and not self.validate_layer_name(return_activations_for_layer):
            print(f"Warning: Layer '{return_activations_for_layer}' not found in YatCNN. Available: {self.get_available_layers()}")
        
        return x
    
    def get_available_layers(self) -> tp.List[str]:
        """Get list of available layer names for activation extraction.
        
        Returns:
            List of layer names that can be used with return_activations_for_layer
        """
        return ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'global_avg_pool']
