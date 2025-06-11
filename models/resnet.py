import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers

# Imports from this project
from aethercv.utils.config import mesh, P, NamedSharding, default_kernel_init, default_bias_init, default_alpha_init

# Custom layer imports
from nmn.nnx.nmn import YatNMN
from nmn.nnx.yatconv import YatConv

Array = jax.Array

class YatResNetBlock(nnx.Module):
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

class LinearResNetBlock(nnx.Module):
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

class BaseResNet(nnx.Module):
    def __init__(self, block_class: tp.Type[YatResNetBlock] | tp.Type[LinearResNetBlock],
                 num_blocks_per_stage: list[int], stage_channels: list[int],
                 num_classes: int, input_channels: int, model_type: str, # "Yat" or "Linear"
                 *, rngs: nnx.Rngs):

        self.model_type = model_type
        conv_kwargs_shared = {
            'kernel_init': nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, None, None, 'model'))),
            'bias_init': nnx.with_partitioning(default_bias_init, NamedSharding(mesh, P('model')))
        }
        if model_type == "Yat":
            conv_kwargs_shared['alpha_init'] = nnx.with_partitioning(default_alpha_init, NamedSharding(mesh, P(None,'model')))

        # Stem
        if model_type == "Yat":
            self.stem_conv = YatConv(
                in_features=input_channels,
                out_features=stage_channels[0],
                kernel_size=(7, 7),
                strides=(2, 2),
                padding='SAME',
                rngs=rngs,
                **conv_kwargs_shared
            )
        else: # LinearResNet
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
        self.max_pool = partial(nnx.avg_pool, window_shape=(3,3), strides=(2,2), padding='SAME')

        # Stages
        self.stages = []
        current_channels = stage_channels[0]
        for i, (num_blocks, channels_out_stage) in enumerate(zip(num_blocks_per_stage, stage_channels)):
            stage_layers = []
            for block_idx in range(num_blocks):
                strides = 2 if block_idx == 0 and i > 0 else 1
                stage_layers.append(block_class(
                    in_channels=current_channels,
                    out_channels=channels_out_stage,
                    strides=strides,
                    rngs=rngs
                ))
                current_channels = channels_out_stage
            self.stages.append(stage_layers) # This should be self.stages.append(nnx.Sequential(*stage_layers)) or similar if you want to treat stages as modules
                                            # For now, keeping as list of layers as in original code structure.

        # Classifier
        if model_type == "Yat":
            self.classifier = YatNMN(
                in_features=current_channels,
                out_features=num_classes,
                rngs=rngs,
                use_bias=False,
                use_alpha=False,
                kernel_init=nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, 'model'))),
                # bias_init and alpha_init are included in YatNMN if use_bias/use_alpha are True
            )
        else: # LinearResNet
            self.classifier = nnx.Linear(
                in_features=current_channels,
                out_features=num_classes,
                rngs=rngs,
                use_bias=False,
                kernel_init=nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, 'model'))),
            )

    def __call__(self, x: Array, training: bool = False, return_activations_for_layer: tp.Optional[str] = None) -> Array:
        activations = {}
        x = self.stem_conv(x); activations['stem_conv_output'] = x
        if return_activations_for_layer == 'stem_conv_output': return x

        if self.model_type == "Linear":
            x = self.stem_bn(x, use_running_average=not training); activations['stem_bn_output'] = x
            if return_activations_for_layer == 'stem_bn_output': return x
            x = nnx.relu(x); activations['stem_relu_output'] = x
            if return_activations_for_layer == 'stem_relu_output': return x

        x = self.max_pool(x); activations['stem_max_pool_output'] = x
        if return_activations_for_layer == 'stem_max_pool_output': return x

        for stage_idx, stage_layers_list in enumerate(self.stages):
            # Assuming stage_layers_list is a list of block modules as constructed
            for block_idx, block in enumerate(stage_layers_list):
                x = block(x, training=training)
                current_activation_key = f'stage{stage_idx}_block{block_idx}_output'
                activations[current_activation_key] = x
                if return_activations_for_layer == current_activation_key:
                    return x

        x = jnp.mean(x, axis=(1, 2)); activations['global_avg_pool'] = x
        if return_activations_for_layer == 'global_avg_pool': return x

        x = self.classifier(x); activations['final_classifier_output'] = x
        if return_activations_for_layer == 'final_classifier_output': return x

        if return_activations_for_layer is not None and return_activations_for_layer not in activations:
            print(f"Warning: Layer '{return_activations_for_layer}' not found in {self.model_type}ResNet. Available: {list(activations.keys())}")
        return x

class YatResNet(BaseResNet):
    def __init__(self, *, num_classes: int, input_channels: int,
                 num_blocks_per_stage: list[int] = [2,2,2,2], # e.g. ResNet18 depth
                 stage_channels: list[int] = [64, 128, 256, 512], # Typical ResNet channels
                 rngs: nnx.Rngs):
        super().__init__(YatResNetBlock, num_blocks_per_stage, stage_channels,
                         num_classes, input_channels, model_type="Yat", rngs=rngs)

class LinearResNet(BaseResNet):
    def __init__(self, *, num_classes: int, input_channels: int,
                 num_blocks_per_stage: list[int] = [2,2,2,2],
                 stage_channels: list[int] = [64, 128, 256, 512],
                 rngs: nnx.Rngs):
        super().__init__(LinearResNetBlock, num_blocks_per_stage, stage_channels,
                         num_classes, input_channels, model_type="Linear", rngs=rngs)
