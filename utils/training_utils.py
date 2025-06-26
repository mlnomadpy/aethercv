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
# Updated to use the new modular structure with factory function
from models import create_model
# Import logging utilities
from utils.logging import create_model_logger, setup_logging, get_global_logger
from config.logging_config import WANDB_CONFIG, MODEL_CONFIGS, DATASET_CONFIGS as LOG_DATASET_CONFIGS

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
    enable_wandb_logging: bool = True,
    experiment_name: str = None,
    wandb_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
):
    print(f"Initializing {model_name} ({model_class_name}) model for dataset {dataset_name} with TPU sharding...")

    config = DATASET_CONFIGS.get(dataset_name)
    
    # Initialize WandB logging if enabled
    logger = None
    if enable_wandb_logging:
        try:
            # Setup WandB configuration
            model_config = {
                "model_class": model_class_name,
                "model_name": model_name,
                "dataset": dataset_name,
                "learning_rate": learning_rate,
                "rng_seed": rng_seed,
                "optimizer": "adamw",
            }
            
            # Add experiment name if provided
            if experiment_name:
                model_config["experiment_name"] = experiment_name
            
            # Add dataset-specific config
            if config:
                model_config.update({
                    "num_classes": config['num_classes'],
                    "input_channels": config['input_channels'],
                    "num_epochs": config['num_epochs'],
                    "batch_size": config['batch_size'],
                })
            
            # Add dataset info from logging config
            if dataset_name in LOG_DATASET_CONFIGS:
                model_config.update(LOG_DATASET_CONFIGS[dataset_name])
            
            # Use WandB config from YAML if provided, otherwise use defaults
            if wandb_config:
                tags = wandb_config.get("tags", [])
                notes = wandb_config.get("notes", "")
                # Override project name if specified in wandb_config but not if global setup was done
                wandb_logger = get_global_logger()
                if not wandb_logger.is_initialized:
                    wandb_logger.project_name = wandb_config.get("project_name", wandb_logger.project_name)
                    wandb_logger.entity = wandb_config.get("entity", wandb_logger.entity)
            else:
                # Get model-specific tags and notes from defaults
                model_key = model_class_name.lower().replace('cnn', '').replace('resnet', '').replace('net', '')
                if model_key in MODEL_CONFIGS:
                    tags = MODEL_CONFIGS[model_key]["tags"]
                    notes = MODEL_CONFIGS[model_key]["notes"]
                else:
                    tags = [model_class_name.lower(), "image-classification"]
                    notes = f"{model_class_name} model training"
            
            # Create model logger
            logger = create_model_logger(model_name, config=model_config)
            
            # Create run name with experiment prefix if provided
            run_name = f"{experiment_name}_{model_name}" if experiment_name else model_name
            
            logger.wandb_logger.init_run(
                model_name=run_name,
                config=model_config,
                tags=tags,
                notes=notes
            )
            print(f"🔗 WandB logging initialized for {model_name}")
            if experiment_name:
                print(f"   Experiment: {experiment_name}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize WandB logging: {e}")
            logger = None

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
        input_channels = ds_info_for_model.features['image'].shape[-1]        
        train_split_name, test_split_name = 'train', 'test'
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

    # Instantiate model using factory function
    model = create_model(
        model_class_name, 
        num_classes=num_classes, 
        input_channels=input_channels, 
        rngs=nnx.Rngs(rng_seed)
    )
    optimizer = nnx.Optimizer(model, optimizer_constructor(learning_rate))
    
    # Log model architecture if WandB is enabled
    if logger:
        try:
            logger.wandb_logger.log_model_architecture(model, model_name)
        except Exception as e:
            print(f"⚠️ Warning: Could not log model architecture: {e}")

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
                
                # Log training metrics to WandB
                if logger:
                    try:
                        logger.log_training_metrics(
                            loss=float(computed_train_metrics.get('loss', 0)),
                            accuracy=float(computed_train_metrics.get('accuracy', 0)),
                            lr=learning_rate
                        )
                    except Exception as e:
                        print(f"⚠️ Warning: Could not log training metrics: {e}")
                
                metrics_computer.reset()

                for test_batch_np in dataset_test_iter.as_numpy_iterator():
                    sharded_test_batch = {
                        k: jax.device_put(jnp.asarray(v), NamedSharding(mesh, P('batch', *(None,) * (jnp.asarray(v).ndim - 1))))
                        for k, v in test_batch_np.items()
                    }
                    eval_step(model, metrics_computer, sharded_test_batch)
                computed_test_metrics = metrics_computer.compute()
                for name, val in computed_test_metrics.items(): metrics_history[f'test_{name}'].append(val)
                
                # Log validation metrics to WandB
                if logger:
                    try:
                        logger.log_validation_metrics(
                            val_loss=float(computed_test_metrics.get('loss', 0)),
                            val_accuracy=float(computed_test_metrics.get('accuracy', 0))
                        )
                        logger.increment_step()
                    except Exception as e:
                        print(f"⚠️ Warning: Could not log validation metrics: {e}")
                
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
    
    # Log final metrics to WandB
    if logger:
        try:
            final_metrics = {
                "final_train_loss": float(metrics_history['train_loss'][-1]) if metrics_history['train_loss'] else 0,
                "final_train_accuracy": float(metrics_history['train_accuracy'][-1]) if metrics_history['train_accuracy'] else 0,
                "final_test_loss": float(metrics_history['test_loss'][-1]) if metrics_history['test_loss'] else 0,
                "final_test_accuracy": float(metrics_history['test_accuracy'][-1]) if metrics_history['test_accuracy'] else 0,
                "total_steps": global_step_counter,
                "epochs_completed": current_num_epochs
            }
            logger.log_custom_metrics(final_metrics)
            print(f"📊 Final metrics logged to WandB for {model_name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not log final metrics: {e}")
    
    metrics_computer.reset()

    print(f"✅ {model_name} Model Training Complete on {dataset_name} after {current_num_epochs} epochs ({global_step_counter} steps)!")
    if metrics_history['test_accuracy'] and metrics_history['test_accuracy'][-1] is not None:
      print(f"   Final Test Accuracy: {metrics_history['test_accuracy'][-1]:.4f}")
    else: print(f"   No test accuracy recorded or available for {model_name}.")
    
    # Finish WandB run
    if logger:
        try:
            logger.wandb_logger.finish_run()
            print(f"🏁 WandB run finished for {model_name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not finish WandB run: {e}")

    return model, metrics_history
