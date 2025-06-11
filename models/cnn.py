import typing as tp
from functools import partial

import jax.numpy as jnp
from flax import nnx

# Imports from this project
# This assumes 'c:\Users\tahab\Documents\github\aetherlm' is in PYTHONPATH
# or the execution context allows 'aethercv' to be resolved as a top-level package.
from aethercv.utils.config import mesh, P, NamedSharding, default_kernel_init, default_bias_init, default_alpha_init

# Custom layer imports (assuming they are in PYTHONPATH or relative)
# Ensure these paths are correct relative to the project structure or PYTHONPATH
from nmn.nnx.nmn import YatNMN
from nmn.nnx.yatconv import YatConv


class YatCNN(nnx.Module):
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
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
        self.dropout4 = nnx.Dropout(rate=0.3, rngs=rngs) # Defined, but dropout3 is used after conv4 in original __call__

        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

        # self.non_linear2 = YatNMN(128, num_classes, use_bias=False, use_alpha=False, rngs=rngs,
        #                           kernel_init=nnx.with_partitioning(default_kernel_init, NamedSharding(mesh, P(None, 'model'))))

    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None):
        activations = {}
        x = self.conv1(x); activations['conv1'] = x
        if return_activations_for_layer == 'conv1': return x
        x = self.dropout1(x, deterministic=not training); x = self.avg_pool(x)

        x = self.conv2(x); activations['conv2'] = x
        if return_activations_for_layer == 'conv2': return x
        x = self.dropout2(x, deterministic=not training); x = self.avg_pool(x)

        x = self.conv3(x); activations['conv3'] = x
        if return_activations_for_layer == 'conv3': return x
        x = self.dropout3(x, deterministic=not training); x = self.avg_pool(x)

        x = self.conv4(x); activations['conv4'] = x
        if return_activations_for_layer == 'conv4': return x
        x = self.dropout3(x, deterministic=not training); x = self.avg_pool(x) # Original train.py uses self.dropout3 here

        x = self.conv5(x); activations['conv5'] = x
        if return_activations_for_layer == 'conv5': return x

        x = jnp.mean(x, axis=(1, 2)); activations['global_avg_pool'] = x
        if return_activations_for_layer == 'global_avg_pool': return x

        # x = self.non_linear2(x); activations['final_layer'] = x
        # if return_activations_for_layer == 'final_layer': return x

        if return_activations_for_layer is not None and return_activations_for_layer not in activations:
            print(f"Warning: Layer '{return_activations_for_layer}' not found in YatCNN. Available: {list(activations.keys())}")
        return x

class LinearCNN(nnx.Module):
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
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
        activations = {}
        x = self.conv1(x); activations['conv1_raw'] = x
        if return_activations_for_layer == 'conv1_raw': return x
        x = nnx.relu(x); activations['conv1'] = x
        if return_activations_for_layer == 'conv1': return x
        x = self.dropout1(x, deterministic=not training); x = self.avg_pool(x)

        x = self.conv2(x); activations['conv2_raw'] = x
        if return_activations_for_layer == 'conv2_raw': return x
        x = nnx.relu(x); activations['conv2'] = x
        if return_activations_for_layer == 'conv2': return x
        x = self.dropout2(x, deterministic=not training); x = self.avg_pool(x)

        x = self.conv3(x); activations['conv3_raw'] = x
        if return_activations_for_layer == 'conv3_raw': return x
        x = nnx.relu(x); activations['conv3'] = x
        if return_activations_for_layer == 'conv3': return x
        x = self.dropout3(x, deterministic=not training); x = self.avg_pool(x)

        x = jnp.mean(x, axis=(1, 2)); activations['global_avg_pool'] = x
        if return_activations_for_layer == 'global_avg_pool': return x

        x = self.linear2(x); activations['final_layer'] = x
        if return_activations_for_layer == 'final_layer': return x

        if return_activations_for_layer is not None and return_activations_for_layer not in activations:
            print(f"Warning: Layer '{return_activations_for_layer}' not found in LinearCNN. Available: {list(activations.keys())}")
        return x
