import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers

# Imports from this project
from utils.config import mesh, P, NamedSharding, default_kernel_init, default_bias_init, default_alpha_init
from models.model import BaseModel

# Custom layer imports
from nmn.nnx.nmn import YatNMN
from nmn.nnx.yatconv import YatConv

Array = jax.Array


class YatResNetBlock(nnx.Module):
    """ResNet block using YAT (Yet Another Transformer) convolution layers."""
    
    def __init__(self, in_channels: int, out_channels: int, strides: int, *, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides

        conv_kwargs_shared = {
            'kernel_init': nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, None, None, 'model'))),
            'bias_init': nnx.with_partitioning(default_bias_init, NamedSharding(mesh, P('model'))),
            'alpha_init': nnx.with_partitioning(default_alpha_init, NamedSharding(mesh, P(None, 'model')))
        }

        self.conv1 = YatConv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(strides, strides),
            padding='SAME',
            rngs=rngs,
            **conv_kwargs_shared
        )
        self.conv2 = YatConv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            rngs=rngs,
            **conv_kwargs_shared
        )
        self.dropout = nnx.Dropout(rate=0.3, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.3, rngs=rngs)

        self.shortcut_projection: tp.Optional[YatConv] = None
        if strides != 1 or in_channels != out_channels:
            self.shortcut_projection = YatConv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                strides=(strides, strides),
                padding='SAME',
                rngs=rngs,
                **conv_kwargs_shared
            )

    def __call__(self, x: Array, training: bool = False) -> Array:
        residual = x
        if self.shortcut_projection is not None:
            residual = self.shortcut_projection(residual)

        x = self.conv1(x)
        x = self.dropout(x, deterministic=not training)
        x = self.conv2(x)
        x = self.dropout2(x, deterministic=not training)

        x += residual
        return x


class YatResNet(BaseModel):
    """YAT ResNet model using custom YAT convolution layers.
    
    This model uses YatResNetBlock which implements residual connections
    with custom YatConv layers for enhanced expressiveness.
    """
    
    def __init__(self, *, num_classes: int, input_channels: int,
                 num_blocks_per_stage: tp.List[int] = [2, 2, 2, 2],  # ResNet18 depth
                 stage_channels: tp.List[int] = [64, 128, 256, 512],  # Typical ResNet channels
                 rngs: nnx.Rngs):
        super().__init__(num_classes=num_classes, input_channels=input_channels, rngs=rngs)
        
        conv_kwargs_shared = {
            'kernel_init': nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, None, None, 'model'))),
            'bias_init': nnx.with_partitioning(default_bias_init, NamedSharding(mesh, P('model'))),
            'alpha_init': nnx.with_partitioning(default_alpha_init, NamedSharding(mesh, P(None, 'model')))
        }

        # Stem
        self.stem_conv = YatConv(
            in_features=input_channels,
            out_features=stage_channels[0],
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='SAME',
            rngs=rngs,
            **conv_kwargs_shared
        )
        self.max_pool = partial(nnx.avg_pool, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # Stages
        self.stages = []
        current_channels = stage_channels[0]
        for i, (num_blocks, channels_out_stage) in enumerate(zip(num_blocks_per_stage, stage_channels)):
            stage_layers = []
            for block_idx in range(num_blocks):
                strides = 2 if block_idx == 0 and i > 0 else 1
                stage_layers.append(YatResNetBlock(
                    in_channels=current_channels,
                    out_channels=channels_out_stage,
                    strides=strides,
                    rngs=rngs
                ))
                current_channels = channels_out_stage
            self.stages.append(stage_layers)

        # Classifier
        self.classifier = YatNMN(
            in_features=current_channels,
            out_features=num_classes,
            rngs=rngs,
            use_bias=False,
            use_alpha=False,
            kernel_init=nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, 'model')))
        )

    def __call__(self, x: Array, training: bool = False, return_activations_for_layer: tp.Optional[str] = None) -> Array:
        """Forward pass through the YAT ResNet model.
        
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
            print(f"Warning: Layer '{return_activations_for_layer}' not found in YatResNet. Available: {self.get_available_layers()}")
        
        return x

    def get_available_layers(self) -> tp.List[str]:
        """Get list of available layer names for activation extraction.
        
        Returns:
            List of layer names that can be used with return_activations_for_layer
        """
        layers = ['stem_conv', 'stem_pool']
        
        # Add stage and block layers
        for stage_idx in range(len(self.stages)):
            for block_idx in range(len(self.stages[stage_idx])):
                layers.append(f'stage{stage_idx}_block{block_idx}')
        
        layers.extend(['global_avg_pool', 'classifier'])
        return layers
