import typing as tp
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

# Imports from this project
from utils.config import (
    DATASET_CONFIGS, GLOBAL_DEFAULT_NUM_EPOCHS, GLOBAL_DEFAULT_EVAL_EVERY, GLOBAL_DEFAULT_BATCH_SIZE, mesh
)
# Import model classes by their names to avoid circular dependencies if utils are imported by models
# This requires models to be available in the python path, e.g. via aethercv.models.cnn or aethercv.models.resnet
from models.cnn import YatCNN, LinearCNN
from models.resnet import YatResNet, LinearResNet

def loss_fn(model, batch):
  # batch['image'] is already sharded if called from train_step/eval_step
  logits = model(batch['image'], training=True) # training flag passed to model
  # batch['label'] is also sharded
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits

@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])
  optimizer.update(grads)

@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, batch):
  def eval_loss_fn(model_eval, batch_eval):
      logits_eval = model_eval(batch_eval['image'], training=False)
      loss_eval = optax.softmax_cross_entropy_with_integer_labels(
          logits=logits_eval, labels=batch_eval['label']
      ).mean()
      return loss_eval, logits_eval

  loss_val, logits_val = eval_loss_fn(model, batch)
  metrics.update(loss=loss_val, logits=logits_val, labels=batch['label'])


def _train_model_loop(
    model_class_name: str,
    model_name: str,
    dataset_name: str,
    rng_seed: int,
    learning_rate: float,
    # momentum: float, # Momentum is often part of optimizer state, not a direct arg to adamw
    optimizer_constructor: tp.Callable[[float], optax.GradientTransformation],
):
    print(f"Initializing {model_name} ({model_class_name}) model for dataset {dataset_name} with TPU sharding...")

    config = DATASET_CONFIGS.get(dataset_name)

    if not config:
        try:
            _, ds_info_for_model = tfds.load(dataset_name, split='train', with_info=True, as_supervised=False)
        except Exception as e:
            print(f"Could not load dataset info for {dataset_name} using tfds.load: {e}. Attempting to proceed with potential errors.")
            class DummyInfo:
                def __init__(self):
                    self.features = {
                        'label': type('DummyLabel', (object,), {'num_classes': 10})(),
                        'image': type('DummyImage', (object,), {'shape': (None, None, 3)})()
                    }
            ds_info_for_model = DummyInfo()

        num_classes = ds_info_for_model.features['label'].num_classes
        input_channels = ds_info_for_model.features['image'].shape[-1]        train_split_name, test_split_name = 'train', 'test'
        # image_key, label_key = 'image', 'label' # These are used in preprocess, ensure consistency
        current_num_epochs = GLOBAL_DEFAULT_NUM_EPOCHS
        current_eval_every = GLOBAL_DEFAULT_EVAL_EVERY
        current_batch_size = GLOBAL_DEFAULT_BATCH_SIZE
        print(f"Warning: Dataset '{dataset_name}' not in pre-defined configs. Inferred info. Using global training params with scaled batch size.")
    else:
        num_classes = config['num_classes']
        input_channels = config['input_channels']
        train_split_name = config['train_split']
        test_split_name = config['test_split']
        # image_key = config['image_key'] # Used in preprocess
        # label_key = config['label_key'] # Used in preprocess
        current_num_epochs = config['num_epochs']
        current_eval_every = config['eval_every']
        current_batch_size = config['batch_size']

    # Instantiate model based on class name
    if model_class_name == "YatCNN":
        model_class_constructor = YatCNN
    elif model_class_name == "LinearCNN":
        model_class_constructor = LinearCNN
    elif model_class_name == "YatResNet":
        model_class_constructor = YatResNet
    elif model_class_name == "LinearResNet":
        model_class_constructor = LinearResNet
    else:
        raise ValueError(f"Unknown model_class_name: {model_class_name}")

    model = model_class_constructor(num_classes=num_classes, input_channels=input_channels, rngs=nnx.Rngs(rng_seed))
    optimizer = nnx.Optimizer(model, optimizer_constructor(learning_rate))

    metrics_computer = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    def augment_data_fn(image_tensor):
        image_tensor = tf.image.random_flip_left_right(image_tensor)
        image_tensor = tf.image.random_brightness(image_tensor, max_delta=0.1)
        image_tensor = tf.image.random_contrast(image_tensor, lower=0.9, upper=1.1)
        return image_tensor

    def preprocess_data_fn(sample, is_training: bool):
        image_key_local = config.get('image_key', 'image') if config else 'image'
        label_key_local = config.get('label_key', 'label') if config else 'label'
        image = tf.cast(sample[image_key_local], tf.float32) / 255.0
        if is_training:
            image = augment_data_fn(image)
        return {'image': image, 'label': sample[label_key_local]}

    loaded_train_ds = tfds.load(dataset_name, split=train_split_name, as_supervised=False, shuffle_files=True)
    loaded_test_ds = tfds.load(dataset_name, split=test_split_name, as_supervised=False)

    dataset_size = loaded_train_ds.cardinality().numpy()
    if dataset_size == tf.data.UNKNOWN_CARDINALITY or dataset_size == tf.data.INFINITE_CARDINALITY:
        print(f"Warning: Dataset size for '{dataset_name}' split '{train_split_name}' is unknown.")
        steps_per_epoch = 50000 // current_batch_size
        if dataset_name in ['cifar10', 'cifar100', 'stl10']:
             steps_per_epoch = { 'cifar10': 50000, 'cifar100': 50000, 'stl10': 5000}.get(dataset_name, 50000) // current_batch_size
    else:
        steps_per_epoch = dataset_size // current_batch_size

    total_expected_steps = current_num_epochs * steps_per_epoch
    print(f"Training {model_name} on {dataset_name} for {current_num_epochs} epochs ({steps_per_epoch} steps/epoch, total ~{total_expected_steps} steps). Global batch size: {current_batch_size}. Eval every {current_eval_every} steps.")

    dataset_test_iter = loaded_test_ds.map(lambda x: preprocess_data_fn(x, is_training=False), num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(current_batch_size, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)

    metrics_history = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}
    global_step_counter = 0

    for epoch in range(current_num_epochs):
        print(f"  Epoch {epoch + 1}/{current_num_epochs}")
        epoch_train_ds = loaded_train_ds.shuffle(buffer_size=max(1024, current_batch_size * 2)) \
            .map(lambda x: preprocess_data_fn(x, is_training=True), num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(current_batch_size, drop_remainder=True) \
            .prefetch(tf.data.AUTOTUNE)

        for batch_in_epoch, batch_data_np in enumerate(epoch_train_ds.as_numpy_iterator()):
            sharded_batch_data = {
                k: jax.device_put(jnp.asarray(v), NamedSharding(mesh, P('batch', *(None,) * (jnp.asarray(v).ndim - 1))))
                for k, v in batch_data_np.items()
            }
            train_step(model, optimizer, metrics_computer, sharded_batch_data)

            if global_step_counter > 0 and (global_step_counter % current_eval_every == 0 or global_step_counter == total_expected_steps -1) and not (epoch == current_num_epochs - 1 and batch_in_epoch == steps_per_epoch -1):
                computed_train_metrics = metrics_computer.compute()
                for name, val in computed_train_metrics.items(): metrics_history[f'train_{name}'].append(val)
                metrics_computer.reset()

                for test_batch_np in dataset_test_iter.as_numpy_iterator():
                    sharded_test_batch = {
                        k: jax.device_put(jnp.asarray(v), NamedSharding(mesh, P('batch', *(None,) * (jnp.asarray(v).ndim - 1))))
                        for k, v in test_batch_np.items()
                    }
                    eval_step(model, metrics_computer, sharded_test_batch)
                computed_test_metrics = metrics_computer.compute()
                for name, val in computed_test_metrics.items(): metrics_history[f'test_{name}'].append(val)
                metrics_computer.reset()
                print(f"    Step {global_step_counter}: {model_name} Train Acc = {metrics_history['train_accuracy'][-1]:.4f}, Test Acc = {metrics_history['test_accuracy'][-1]:.4f}")

            global_step_counter += 1
            if global_step_counter >= total_expected_steps: break
        if global_step_counter >= total_expected_steps: break

    print(f"  Performing final evaluation for {model_name} after {current_num_epochs} epochs...")
    computed_train_metrics = metrics_computer.compute() # Compute any remaining train metrics
    if computed_train_metrics and 'loss' in computed_train_metrics :
        for name, val in computed_train_metrics.items(): metrics_history[f'train_{name}'].append(val)
    metrics_computer.reset()

    for test_batch_np in dataset_test_iter.as_numpy_iterator():
        sharded_test_batch = {
            k: jax.device_put(jnp.asarray(v), NamedSharding(mesh, P('batch', *(None,) * (jnp.asarray(v).ndim - 1))))
            for k, v in test_batch_np.items()
        }
        eval_step(model, metrics_computer, sharded_test_batch)
    computed_test_metrics = metrics_computer.compute()
    for name, val in computed_test_metrics.items(): metrics_history[f'test_{name}'].append(val)
    metrics_computer.reset()

    print(f"✅ {model_name} Model Training Complete on {dataset_name} after {current_num_epochs} epochs ({global_step_counter} steps)!")
    if metrics_history['test_accuracy'] and metrics_history['test_accuracy'][-1] is not None:
      print(f"   Final Test Accuracy: {metrics_history['test_accuracy'][-1]:.4f}")
    else: print(f"   No test accuracy recorded or available for {model_name}.")

    return model, metrics_history
