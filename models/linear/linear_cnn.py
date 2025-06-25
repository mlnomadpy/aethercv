import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

# Imports from this project
from utils.config import mesh, P, NamedSharding, default_kernel_init, default_bias_init
from models.model import BaseModel


class LinearCNN(BaseModel):
    """Linear CNN model using standard convolution layers.
    
    This model uses standard Conv layers with ReLU activations
    and includes a final linear layer for classification.
    """
    
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        super().__init__(num_classes=num_classes, input_channels=input_channels, rngs=rngs)
        
        conv_kwargs = {
            'kernel_init': nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, None, None, 'model'))),
            'bias_init': nnx.with_partitioning(default_bias_init, NamedSharding(mesh, P('model')))
        }
        
        self.conv1 = nnx.Conv(input_channels, 32, kernel_size=(5, 5), rngs=rngs, **conv_kwargs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(5, 5), rngs=rngs, **conv_kwargs)
        self.conv3 = nnx.Conv(64, 128, kernel_size=(5, 5), rngs=rngs, **conv_kwargs)

        self.dropout1 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout3 = nnx.Dropout(rate=0.3, rngs=rngs)

        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear2 = nnx.Linear(128, num_classes, rngs=rngs, use_bias=False,
                                  kernel_init=nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, 'model'))))

    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None):
        """Forward pass through the Linear CNN model.
        
        Args:
            x: Input tensor of shape (batch, height, width, channels)
            training: Whether the model is in training mode
            return_activations_for_layer: Optional layer name to return activations for
            
        Returns:
            Model output or activations for specified layer
        """
        activations = {}
        
        x = self.conv1(x)
        activations['conv1_raw'] = x
        if return_activations_for_layer == 'conv1_raw': 
            return x
        x = nnx.relu(x)
        activations['conv1'] = x
        if return_activations_for_layer == 'conv1': 
            return x
        x = self.dropout1(x, deterministic=not training)
        x = self.avg_pool(x)

        x = self.conv2(x)
        activations['conv2_raw'] = x
        if return_activations_for_layer == 'conv2_raw': 
            return x
        x = nnx.relu(x)
        activations['conv2'] = x
        if return_activations_for_layer == 'conv2': 
            return x
        x = self.dropout2(x, deterministic=not training)
        x = self.avg_pool(x)

        x = self.conv3(x)
        activations['conv3_raw'] = x
        if return_activations_for_layer == 'conv3_raw': 
            return x
        x = nnx.relu(x)
        activations['conv3'] = x
        if return_activations_for_layer == 'conv3': 
            return x
        x = self.dropout3(x, deterministic=not training)
        x = self.avg_pool(x)

        x = jnp.mean(x, axis=(1, 2))
        activations['global_avg_pool'] = x
        if return_activations_for_layer == 'global_avg_pool': 
            return x

        x = self.linear2(x)
        activations['final_layer'] = x
        if return_activations_for_layer == 'final_layer': 
            return x

        if return_activations_for_layer is not None and not self.validate_layer_name(return_activations_for_layer):
            print(f"Warning: Layer '{return_activations_for_layer}' not found in LinearCNN. Available: {self.get_available_layers()}")
        
        return x
    
    def get_available_layers(self) -> tp.List[str]:
        """Get list of available layer names for activation extraction.
        
        Returns:
            List of layer names that can be used with return_activations_for_layer
        """
        return ['conv1_raw', 'conv1', 'conv2_raw', 'conv2', 'conv3_raw', 'conv3', 'global_avg_pool', 'final_layer']
