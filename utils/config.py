import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from flax.nnx.nn import initializers

# Initialize JAX devices and mesh
jax.devices()
# Create a `Mesh` object representing TPU device arrangement.
device_mesh_shape = (4, 2) 
try:
    mesh_devices = mesh_utils.create_device_mesh(device_mesh_shape)
    print(f"Successfully created {device_mesh_shape} device mesh.")
except Exception as e:
    print(f"Could not create {device_mesh_shape} device mesh: {e}. Falling back to all available devices.")
    available_devices = jax.devices()
    num_devices = len(available_devices)
    if num_devices >= 2 and num_devices % 2 == 0:
        fallback_shape = (num_devices // 2, 2)
    else:
        fallback_shape = (num_devices, 1)
    try:
        mesh_devices = mesh_utils.create_device_mesh(fallback_shape)
        print(f"Using fallback device mesh shape: {fallback_shape}")
    except Exception as e_fallback:
        print(f"Could not create fallback mesh {fallback_shape}: {e_fallback}. Using a single device mesh.")
        mesh_devices = mesh_utils.create_device_mesh((1,1)) 
        print("Using single device mesh (1,1). TPU sharding will be minimal.")

mesh = Mesh(mesh_devices, ('batch', 'model'))
Array = jax.Array

# Default initializers
default_kernel_init = initializers.kaiming_normal()
default_bias_init = initializers.zeros_init()
default_alpha_init = initializers.ones_init()

# ===== DATASET CONFIGURATIONS =====
DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10, 'input_channels': 3,
        'train_split': 'train', 'test_split': 'test',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 5, 'eval_every': 200, 'batch_size': 64 * device_mesh_shape[0] 
    },
    'cifar100': {
        'num_classes': 100, 'input_channels': 3,
        'train_split': 'train', 'test_split': 'test',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 5, 'eval_every': 200, 'batch_size': 64 * device_mesh_shape[0]
    },
    'stl10': {
        'num_classes': 10, 'input_channels': 3,
        'train_split': 'train', 'test_split': 'test',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 200, 'eval_every': 200, 'batch_size': 128 * device_mesh_shape[0]
    },
    'eurosat/rgb': {
        'num_classes': 10, 'input_channels': 3,
        'train_split': 'train[:80%]', 'test_split': 'train[80%:]',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 20, 'eval_every': 100, 'batch_size': 32 * device_mesh_shape[0]
    },
    'eurosat/all': {
        'num_classes': 10, 'input_channels': 13,
        'train_split': 'train[:80%]', 'test_split': 'train[80%:]',
        'image_key': 'image', 'label_key': 'label',
        'num_epochs': 20, 'eval_every': 100, 'batch_size': 16 * device_mesh_shape[0]
    },
}

# Global defaults if a dataset is not in DATASET_CONFIGS
# These are used by _train_model_loop if a dataset_name is not found in DATASET_CONFIGS
_DEFAULT_DATASET_FOR_GLOBALS = 'cifar10' 
_default_config_for_globals = DATASET_CONFIGS.get(_DEFAULT_DATASET_FOR_GLOBALS, {})
GLOBAL_DEFAULT_NUM_EPOCHS = _default_config_for_globals.get('num_epochs', 10)
GLOBAL_DEFAULT_EVAL_EVERY = _default_config_for_globals.get('eval_every', 200)
GLOBAL_DEFAULT_BATCH_SIZE = _default_config_for_globals.get('batch_size', 64 * device_mesh_shape[0])