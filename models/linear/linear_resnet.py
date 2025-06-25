import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers

# Imports from this project
from utils.config import mesh, P, NamedSharding, default_kernel_init, default_bias_init
from models.model import BaseModel

Array = jax.Array


class LinearResNetBlock(nnx.Module):
    """ResNet block using standard linear convolution layers with batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, strides: int, *, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides

        conv_kwargs_shared = {
            'kernel_init': nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, None, None, 'model'))),
            'bias_init': nnx.with_partitioning(default_bias_init, NamedSharding(mesh, P('model')))
        }
        norm_kwargs_shared = {
            'scale_init': nnx.with_partitioning(initializers.ones_init(), NamedSharding(mesh, P('model'))),
            'bias_init': nnx.with_partitioning(initializers.zeros_init(), NamedSharding(mesh, P('model')))
        }

        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(strides, strides),
            padding='SAME',
            rngs=rngs,
            **conv_kwargs_shared
        )
        self.bn1 = nnx.BatchNorm(num_features=out_channels, rngs=rngs, **norm_kwargs_shared)

        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            rngs=rngs,
            **conv_kwargs_shared
        )
        self.bn2 = nnx.BatchNorm(num_features=out_channels, rngs=rngs, **norm_kwargs_shared)
        self.dropout = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.3, rngs=rngs)

        self.shortcut_projection: tp.Optional[nnx.Conv] = None
        self.shortcut_bn: tp.Optional[nnx.BatchNorm] = None
        if strides != 1 or in_channels != out_channels:
            self.shortcut_projection = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                strides=(strides, strides),
                padding='SAME',
                rngs=rngs,
                **conv_kwargs_shared
            )
            self.shortcut_bn = nnx.BatchNorm(num_features=out_channels, rngs=rngs, **norm_kwargs_shared)

    def __call__(self, x: Array, training: bool = False) -> Array:
        residual = x
        if self.shortcut_projection is not None and self.shortcut_bn is not None:
            residual = self.shortcut_projection(residual)
            residual = self.shortcut_bn(residual, use_running_average=not training)

        y = self.conv1(x)
        y = self.bn1(y, use_running_average=not training)
        y = nnx.relu(y)
        y = self.dropout(y, deterministic=not training)

        y = self.conv2(y)
        y = self.bn2(y, use_running_average=not training)
        y = self.dropout2(y, deterministic=not training)

        y += residual
        y = nnx.relu(y)
        return y


class LinearResNet(BaseModel):
    """Linear ResNet model using standard convolution layers with batch normalization.
    
    This model uses LinearResNetBlock which implements residual connections
    with standard Conv layers and batch normalization.
    """
    
    def __init__(self, *, num_classes: int, input_channels: int,
                 num_blocks_per_stage: tp.List[int] = [2, 2, 2, 2],  # ResNet18 depth
                 stage_channels: tp.List[int] = [64, 128, 256, 512],  # Typical ResNet channels
                 rngs: nnx.Rngs):
        super().__init__(num_classes=num_classes, input_channels=input_channels, rngs=rngs)
        
        conv_kwargs_shared = {
            'kernel_init': nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, None, None, 'model'))),
            'bias_init': nnx.with_partitioning(default_bias_init, NamedSharding(mesh, P('model')))
        }

        # Stem
        self.stem_conv = nnx.Conv(
            in_features=input_channels,
            out_features=stage_channels[0],
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='SAME',
            rngs=rngs,
            **conv_kwargs_shared
        )
        self.stem_bn = nnx.BatchNorm(
            num_features=stage_channels[0],
            rngs=rngs,
            scale_init=nnx.with_partitioning(initializers.ones_init(), NamedSharding(mesh, P('model'))),
            bias_init=nnx.with_partitioning(initializers.zeros_init(), NamedSharding(mesh, P('model')))
        )
        self.max_pool = partial(nnx.avg_pool, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # Stages
        self.stages = []
        current_channels = stage_channels[0]
        for i, (num_blocks, channels_out_stage) in enumerate(zip(num_blocks_per_stage, stage_channels)):
            stage_layers = []
            for block_idx in range(num_blocks):
                strides = 2 if block_idx == 0 and i > 0 else 1
                stage_layers.append(LinearResNetBlock(
                    in_channels=current_channels,
                    out_channels=channels_out_stage,
                    strides=strides,
                    rngs=rngs
                ))
                current_channels = channels_out_stage
            self.stages.append(stage_layers)

        # Classifier
        self.classifier = nnx.Linear(
            in_features=current_channels,
            out_features=num_classes,
            rngs=rngs,
            use_bias=False,
            kernel_init=nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, 'model')))
        )

    def __call__(self, x: Array, training: bool = False, return_activations_for_layer: tp.Optional[str] = None) -> Array:
        """Forward pass through the Linear ResNet model.
        
        Args:
            x: Input tensor of shape (batch, height, width, channels)
            training: Whether the model is in training mode
            return_activations_for_layer: Optional layer name to return activations for
            
        Returns:
            Model output or activations for specified layer
        """
        activations = {}
        
        x = self.stem_conv(x)
        activations['stem_conv'] = x
        if return_activations_for_layer == 'stem_conv': 
            return x

        x = self.stem_bn(x, use_running_average=not training)
        activations['stem_bn'] = x
        if return_activations_for_layer == 'stem_bn': 
            return x
        
        x = nnx.relu(x)
        activations['stem_relu'] = x
        if return_activations_for_layer == 'stem_relu': 
            return x

        x = self.max_pool(x)
        activations['stem_pool'] = x
        if return_activations_for_layer == 'stem_pool': 
            return x

        for stage_idx, stage_layers_list in enumerate(self.stages):
            for block_idx, block in enumerate(stage_layers_list):
                x = block(x, training=training)
                current_activation_key = f'stage{stage_idx}_block{block_idx}'
                activations[current_activation_key] = x
                if return_activations_for_layer == current_activation_key:
                    return x

        x = jnp.mean(x, axis=(1, 2))
        activations['global_avg_pool'] = x
        if return_activations_for_layer == 'global_avg_pool': 
            return x

        x = self.classifier(x)
        activations['classifier'] = x
        if return_activations_for_layer == 'classifier': 
            return x

        if return_activations_for_layer is not None and not self.validate_layer_name(return_activations_for_layer):
            print(f"Warning: Layer '{return_activations_for_layer}' not found in LinearResNet. Available: {self.get_available_layers()}")
        
        return x

    def get_available_layers(self) -> tp.List[str]:
        """Get list of available layer names for activation extraction.
        
        Returns:
            List of layer names that can be used with return_activations_for_layer
        """
        layers = ['stem_conv', 'stem_bn', 'stem_relu', 'stem_pool']
        
        # Add stage and block layers
        for stage_idx in range(len(self.stages)):
            for block_idx in range(len(self.stages[stage_idx])):
                layers.append(f'stage{stage_idx}_block{block_idx}')
        
        layers.extend(['global_avg_pool', 'classifier'])
        return layers
