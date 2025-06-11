import typing as tp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
import os
from datetime import datetime

# Assuming test_ds_iter is a tf.data.Dataset iterator that yields batches of numpy arrays
# and class_names is a list of strings.

Array = jax.Array # For type hinting if JAX arrays are directly handled

# Create directory for saving visualizations
def create_viz_directory():
    """Create a directory for saving visualizations with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = f"visualizations_{timestamp}"
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir

def compare_training_curves(histories_list: list[dict], model_names_list: list[str], save_dir: str = None):
    num_models = len(histories_list)
    if num_models == 0 or len(model_names_list) != num_models:
        print("Error: Invalid input for compare_training_curves. Ensure histories_list and model_names_list are non-empty and have matching lengths.")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Training Curves Comparison: All Models', fontsize=16, fontweight='bold')

    colors = plt.cm.get_cmap('tab10', max(10, num_models)).colors
    line_styles = ['-', '--', '-.', ':'] * (num_models // 4 + 1)

    min_steps = float('inf')
    for history in histories_list:
        if history.get('train_loss'):
            min_steps = min(min_steps, len(history['train_loss']))
        else:
            print(f"Warning: A model's history is missing 'train_loss' or is empty. Skipping this model for step calculation.")

    if min_steps == float('inf') or min_steps == 0:
        print("Error: Could not determine valid number of steps from histories. Aborting curve plotting.")
        plt.close(fig)
        return

    steps_range = range(min_steps)

    for i in range(num_models):
        history = histories_list[i]
        model_name = model_names_list[i]
        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]

        if history.get('train_loss') and len(history['train_loss']) >= min_steps:
            ax1.plot(steps_range, history['train_loss'][:min_steps], color=color, linestyle=linestyle, label=model_name, linewidth=2)
        if history.get('test_loss') and len(history['test_loss']) >= min_steps:
            ax2.plot(steps_range, history['test_loss'][:min_steps], color=color, linestyle=linestyle, label=model_name, linewidth=2)
        if history.get('train_accuracy') and len(history['train_accuracy']) >= min_steps:
            ax3.plot(steps_range, history['train_accuracy'][:min_steps], color=color, linestyle=linestyle, label=model_name, linewidth=2)
        if history.get('test_accuracy') and len(history['test_accuracy']) >= min_steps:
            ax4.plot(steps_range, history['test_accuracy'][:min_steps], color=color, linestyle=linestyle, label=model_name, linewidth=2)    
            ax1.set_title('Training Loss', fontweight='bold'); ax1.set_xlabel('Evaluation Steps'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title('Test Loss', fontweight='bold'); ax2.set_xlabel('Evaluation Steps'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax3.set_title('Training Accuracy', fontweight='bold'); ax3.set_xlabel('Evaluation Steps'); ax3.set_ylabel('Accuracy'); ax3.legend(); ax3.grid(True, alpha=0.3)
    ax4.set_title('Test Accuracy', fontweight='bold'); ax4.set_xlabel('Evaluation Steps'); ax4.set_ylabel('Accuracy'); ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    if save_dir is None:
        save_dir = create_viz_directory()
    
    training_curves_dir = os.path.join(save_dir, "training_curves")
    os.makedirs(training_curves_dir, exist_ok=True)
    filename = "training_curves_comparison.png"
    filepath = os.path.join(training_curves_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Training curves comparison saved to: {filepath}")

def print_final_metrics_comparison_all(histories_list: list[dict], model_names_list: list[str]):
    num_models = len(histories_list)
    if num_models == 0 or len(model_names_list) != num_models:
        print("Error: Invalid input for print_final_metrics_comparison_all.")
        return

    print("\n📊 FINAL METRICS COMPARISON (ALL MODELS)"); print("=" * (20 + num_models * 18))

    final_metrics_data = []
    for history in histories_list:
        final_metrics_data.append({k: v[-1] for k, v in history.items() if v and isinstance(v, list) and len(v) > 0})

    header = f"{'Metric':<20}" + "".join([f"{name:<18}" for name in model_names_list])
    print(header)
    print("-" * len(header))

    metrics_to_display = ['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']
    best_test_accuracy = -1.0
    best_model_name_acc = "N/A"

    for metric_key in metrics_to_display:
        row_str = f"{metric_key:<20}"
        for i in range(num_models):
            val = final_metrics_data[i].get(metric_key, float('nan'))
            row_str += f"{val:<18.4f}"
            if metric_key == 'test_accuracy' and val > best_test_accuracy:
                best_test_accuracy = val
                best_model_name_acc = model_names_list[i]
        print(row_str)

    print("\n🏆 OVERALL SUMMARY (Based on Test Accuracy):")
    if best_test_accuracy != -1.0:
        print(f"   🥇 Best Test Accuracy: {best_model_name_acc} ({best_test_accuracy:.4f})")
        for i in range(num_models):
            model_name = model_names_list[i]
            acc = final_metrics_data[i].get('test_accuracy', float('nan'))
            print(f"     - {model_name}: {acc:.4f}")
    else:
        print("   Could not determine the best model (no valid test accuracy found).")

def analyze_convergence_all(histories_list: list[dict], model_names_list: list[str]):
    num_models = len(histories_list)
    if num_models == 0 or len(model_names_list) != num_models:
        print("Error: Invalid input for analyze_convergence_all.")
        return

    print("\n🔍 CONVERGENCE ANALYSIS (ALL MODELS)"); print("=" * (25 + num_models * 18))

    def calculate_convergence_metrics(history):
        if not all(k in history and history[k] for k in ['test_accuracy', 'train_accuracy', 'test_loss']):
            return {'convergence_step': -1, 'stability': float('nan'), 'overfitting': float('nan'), 'final_loss': float('nan')}

        test_acc = history['test_accuracy']
        train_acc = history['train_accuracy']
        test_loss = history['test_loss']

        if not test_acc:
            final_loss_val_empty_acc = history.get('test_loss', [float('nan')])[-1] if history.get('test_loss') else float('nan')
            return {'convergence_step': -1, 'stability': float('nan'), 'overfitting': float('nan'), 'final_loss': final_loss_val_empty_acc}

        final_acc_val = test_acc[-1]
        target_acc = 0.5 * final_acc_val
        convergence_step = next((i for i, acc in enumerate(test_acc) if acc >= target_acc), len(test_acc) - 1 if test_acc else -1)

        last_quarter_start_idx = len(test_acc) - (len(test_acc) // 4)
        stability = np.std(test_acc[last_quarter_start_idx:]) if len(test_acc) // 4 > 0 and len(test_acc[last_quarter_start_idx:]) > 0 else 0.0

        overfitting = (train_acc[-1] if train_acc else float('nan')) - final_acc_val
        final_loss_val = test_loss[-1] if test_loss else float('nan')

        return {
            'convergence_step': convergence_step,
            'stability': stability,
            'overfitting': overfitting,
            'final_loss': final_loss_val
        }

    all_convergence_data = []
    for history in histories_list:
        all_convergence_data.append(calculate_convergence_metrics(history))

    header = f"{'Metric':<25}" + "".join([f"{name:<18}" for name in model_names_list])
    print(header)
    print("-" * len(header))

    metrics_to_display = ['convergence_step', 'stability', 'overfitting', 'final_loss']
    fastest_convergence_step = float('inf')
    fastest_model_name_conv = "N/A"

    for metric_key in metrics_to_display:
        row_str = f"{metric_key.replace('_', ' ').title():<25}"
        for i in range(num_models):
            val = all_convergence_data[i].get(metric_key, float('nan'))
            row_str += f"{val:<18.4f}"
            if metric_key == 'convergence_step' and val != -1 and val < fastest_convergence_step:
                fastest_convergence_step = val
                fastest_model_name_conv = model_names_list[i]
        print(row_str)

    print("\n📋 CONVERGENCE SUMMARY:")
    if fastest_convergence_step != float('inf') and fastest_model_name_conv != "N/A":
        print(f"   🚀 Fastest Convergence (to 50% of final test acc): {fastest_model_name_conv} (at step {int(fastest_convergence_step)})")
    else:
        print("   Could not determine the fastest converging model (or no model converged meaningfully).")

def detailed_test_evaluation_all(models_list: list[nnx.Module], model_names_list: list[str], test_ds_iter, class_names: list[str]):
    num_models = len(models_list)
    if num_models == 0 or len(model_names_list) != num_models:
        print("Error: Invalid input for detailed_test_evaluation_all.")
        return {}

    print(f"\n🎯 DETAILED TEST EVALUATION (ALL {num_models} MODELS)"); print("=" * 70)

    num_classes = len(class_names)
    all_predictions_dict = {name: [] for name in model_names_list}
    true_labels_list = []

    print(f"Running predictions for {num_models} models...")
    for batch_idx, batch in enumerate(test_ds_iter.as_numpy_iterator()): # Assuming test_ds_iter yields numpy batches
        batch_images_local = batch['image']
        batch_labels_local = batch['label']
        true_labels_list.extend(np.array(batch_labels_local).tolist())

        for i in range(num_models):
            model = models_list[i]
            model_name = model_names_list[i]
            # Assuming model can take JAX array if it's a JAX model, or numpy if TF/Keras
            # If models are JAX/Flax, ensure batch_images_local is jnp.asarray if needed by model
            # For this example, assuming model call is compatible with numpy input from tf.data iterator
            logits = model(jnp.asarray(batch_images_local), training=False) # Pass JAX array
            preds = jnp.argmax(logits, axis=1)
            all_predictions_dict[model_name].extend(np.array(preds).tolist())

    true_labels_np = np.array(true_labels_list)
    for name in model_names_list:
        all_predictions_dict[name] = np.array(all_predictions_dict[name])

    print("\n🎯 PER-CLASS ACCURACY COMPARISON (ALL MODELS)");
    header = f"{'Class':<15}" + "".join([f"{name[:10]+'.':<12}" for name in model_names_list]) + f"{'Samples':<10}"
    print(header); print("-" * len(header))

    for class_idx in range(num_classes):
        class_mask = (true_labels_np == class_idx)
        class_samples = np.sum(class_mask)
        row_str = f"{class_names[class_idx]:<15}"
        if class_samples > 0:
            for model_name in model_names_list:
                model_preds_for_class = all_predictions_dict[model_name][class_mask]
                true_labels_for_class = true_labels_np[class_mask]
                acc = np.mean(model_preds_for_class == true_labels_for_class)
                row_str += f"{acc:<12.4f}"
            row_str += f"{int(class_samples):<10}"
        else:
            for _ in model_names_list:
                row_str += f"{'N/A':<12}"
            row_str += f"{int(class_samples):<10}"
        print(row_str)

    print(f"\n🤝 MODEL AGREEMENT ANALYSIS (ALL {num_models} MODELS)"); print("=" * 50)
    if num_models > 1:
        first_model_preds = all_predictions_dict[model_names_list[0]]
        all_agree_mask = np.ones_like(first_model_preds, dtype=bool)
        for i in range(1, num_models):
            all_agree_mask &= (all_predictions_dict[model_names_list[i]] == first_model_preds)
        all_models_agree_perc = np.mean(all_agree_mask) * 100
        print(f"All {num_models} models agree on prediction: {all_models_agree_perc:.2f}%")

        all_correct_mask = np.ones_like(first_model_preds, dtype=bool)
        for model_name in model_names_list:
            all_correct_mask &= (all_predictions_dict[model_name] == true_labels_np)
        all_models_correct_perc = np.mean(all_correct_mask) * 100
        print(f"All {num_models} models are correct: {all_models_correct_perc:.2f}%")
    else:
        model_acc = np.mean(all_predictions_dict[model_names_list[0]] == true_labels_np) * 100
        print(f"Accuracy for the single model ({model_names_list[0]}): {model_acc:.2f}%")

    predictions_package = {
        'all_predictions': all_predictions_dict,
        'true_labels': true_labels_np,
        'class_names': class_names,
        'model_names': model_names_list
    }
    if num_models > 1:
      predictions_package['all_models_agree_perc'] = all_models_agree_perc
      predictions_package['all_models_correct_perc'] = all_models_correct_perc

    return predictions_package

def plot_confusion_matrices_all(predictions_package: dict, save_dir: str = None):
    all_predictions = predictions_package.get('all_predictions')
    true_labels = predictions_package.get('true_labels')
    class_names = predictions_package.get('class_names')
    model_names = predictions_package.get('model_names')

    if not all_predictions or true_labels is None or not class_names or not model_names:
        print("Error: Missing data in predictions_package for plot_confusion_matrices_all.")
        return

    num_models = len(model_names)
    if num_models == 0:
        print("No models to plot confusion matrices for.")
        return

    print(f"\n📊 PLOTTING CONFUSION MATRICES (ALL {num_models} MODELS)"); print("=" * 50)

    cols = int(np.ceil(np.sqrt(num_models)))
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
    axes_flat = axes.flatten()

    for i, model_name in enumerate(model_names):
        if model_name not in all_predictions:
            print(f"Warning: Predictions for model '{model_name}' not found in predictions_package.")
            if i < len(axes_flat):
                axes_flat[i].set_title(f'{model_name} - No Data')
                axes_flat[i].axis('off')
            continue        
        model_preds = all_predictions[model_name]
        cm = confusion_matrix(true_labels, model_preds)

        ax = axes_flat[i]
        # Replace seaborn heatmap with matplotlib imshow
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{model_name} - CM', fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Add text annotations
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, str(cm[j, k]), ha='center', va='center', 
                       color='white' if cm[j, k] > cm.max() / 2 else 'black')
        
        # Set tick labels
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)    
        for j in range(num_models, len(axes_flat)):
            fig.delaxes(axes_flat[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'Confusion Matrices for All {num_models} Models', fontsize=16, fontweight='bold')
    
    # Save the plot
    if save_dir is None:
        save_dir = create_viz_directory()
    
    confusion_matrix_dir = os.path.join(save_dir, "confusion_matrices")
    os.makedirs(confusion_matrix_dir, exist_ok=True)
    filename = "confusion_matrices_all_models.png"
    filepath = os.path.join(confusion_matrix_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrices saved to: {filepath}")

def generate_summary_report_all(histories_list: list[dict], model_names_list: list[str], predictions_package: dict):
    num_models = len(histories_list)
    if num_models == 0 or len(model_names_list) != num_models:
        print("Error: Invalid input for generate_summary_report_all (histories/model names).")
        return
    if not predictions_package or 'all_predictions' not in predictions_package:
        print("Warning: Missing or incomplete predictions_package for generate_summary_report_all. Report will be partial.")
        predictions_package = {} # Ensure it's a dict to avoid errors below

    print("\n" + "="*80 + "\n                    COMPREHENSIVE SUMMARY REPORT (ALL MODELS)\n" + "="*80)

    final_metrics_data = []
    for history in histories_list:
        final_metrics_data.append({k: v[-1] if isinstance(v, list) and v else float('nan')
                                   for k, v in history.items()})

    best_test_accuracy = -1.0
    best_model_name_acc = "N/A"
    for i in range(num_models):
        val = final_metrics_data[i].get('test_accuracy', float('nan'))
        if not np.isnan(val) and val > best_test_accuracy:
            best_test_accuracy = val
            best_model_name_acc = model_names_list[i]

    print(f"\n🏆 OVERALL WINNER (Based on Highest Test Accuracy):")
    if best_model_name_acc != "N/A":
        print(f"   🥇 {best_model_name_acc} wins with {best_test_accuracy:.4f} test accuracy!")
    else:
        print(f"   Could not determine a clear winner based on test accuracy.")

    print(f"\n📈 PERFORMANCE SUMMARY (Final Test Accuracies):")
    for i in range(num_models):
        model_name = model_names_list[i]
        acc = final_metrics_data[i].get('test_accuracy', float('nan'))
        loss = final_metrics_data[i].get('test_loss', float('nan'))
        train_acc = final_metrics_data[i].get('train_accuracy', float('nan'))
        train_loss = final_metrics_data[i].get('train_loss', float('nan'))
        print(f"   - {model_name}: Test Acc: {acc:.4f}, Test Loss: {loss:.4f} (Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f})")

    if num_models > 1 and predictions_package and predictions_package.get('all_models_agree_perc') is not None:
        agree_perc = predictions_package['all_models_agree_perc']
        correct_perc = predictions_package['all_models_correct_perc']
        print(f"\n🤝 MODEL AGREEMENT (among all {num_models} models on test set predictions):")
        print(f"   All models agree on prediction: {agree_perc:.2f}%")
        print(f"   All models are correct: {correct_perc:.2f}%")

    fastest_convergence_step = float('inf')
    fastest_model_name_conv = "N/A"
    lowest_test_loss = float('inf')
    lowest_loss_model_name = "N/A"

    for i, history in enumerate(histories_list):
        model_name = model_names_list[i]
        if history.get('test_accuracy') and isinstance(history['test_accuracy'], list) and history['test_accuracy']:
            test_acc_hist = history['test_accuracy']
            final_acc_val_conv = test_acc_hist[-1]
            if not np.isnan(final_acc_val_conv) and final_acc_val_conv > 0:
                target_acc_conv = 0.5 * final_acc_val_conv
                current_conv_step = next((step_idx for step_idx, acc_val in enumerate(test_acc_hist) if acc_val >= target_acc_conv), -1)
                if current_conv_step != -1 and current_conv_step < fastest_convergence_step:
                    fastest_convergence_step = current_conv_step
                    fastest_model_name_conv = model_name

        final_loss = final_metrics_data[i].get('test_loss', float('inf'))
        if not np.isnan(final_loss) and final_loss < lowest_test_loss:
            lowest_test_loss = final_loss
            lowest_loss_model_name = model_name

    print(f"\n⭐ OTHER HIGHLIGHTS:")
    if fastest_model_name_conv != "N/A":
        print(f"   🚀 Fastest Convergence (to 50% of its final test acc): {fastest_model_name_conv} (at evaluation step {fastest_convergence_step})")
    else:
        print("   Could not determine fastest converging model from available history.")

    if lowest_loss_model_name != "N/A":
        print(f"   📉 Lowest Final Test Loss: {lowest_loss_model_name} ({lowest_test_loss:.4f})")
    else:
        print("   Could not determine model with the lowest test loss.")

    print("="*80)

def visualize_kernels_all(models_list: list[nnx.Module], model_names_list: list[str], layer_name_map: dict[str, str], num_kernels_to_show=16, save_dir: str = None):
    num_models = len(models_list)
    if num_models == 0 or len(model_names_list) != num_models or len(layer_name_map) != num_models:
        print("Error: Invalid input for visualize_kernels_all.")
        return

    print(f"\n🎨 VISUALIZING KERNELS FOR ALL {num_models} MODELS"); print("=" * 50)

    def get_kernels_from_model(model, layer_name_str):
        try:
            layer = model
            parts = layer_name_str.split('.')
            for part in parts:
                if not hasattr(layer, part):
                    print(f"Layer part '{part}' not found in '{layer_name_str}' for model {model.__class__.__name__}")
                    return None
                layer = getattr(layer, part)

            if hasattr(layer, 'kernel') and hasattr(layer.kernel, 'value'):
                return layer.kernel.value
            elif isinstance(layer, nnx.Param):
                 return layer.value
            print(f"Kernel attribute .kernel.value not found in layer '{layer_name_str}' of {model.__class__.__name__}.")
            return None
        except AttributeError as e:
            print(f"AttributeError while accessing layer '{layer_name_str}' in {model.__class__.__name__}: {e}")
            return None
        except Exception as e_general:
            print(f"General error accessing layer '{layer_name_str}' in {model.__class__.__name__}: {e_general}")
            return None

    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        layer_name = layer_name_map.get(model_name)

        if not layer_name:
            print(f"Warning: Layer name for kernel visualization not specified for model '{model_name}'. Skipping.")
            continue

        print(f"\n🖼️ Visualizing kernels from layer '{layer_name}' for model '{model_name}'...")
        kernels = get_kernels_from_model(model, layer_name)

        if kernels is None:
            print(f"Could not retrieve kernels for '{model_name}', layer '{layer_name}'.")
            continue

        try:
            kernels_np = np.array(kernels)
            if kernels_np.ndim not in [2, 3, 4]:
                print(f"Unsupported kernel dimensions ({kernels_np.ndim}) for '{model_name}' layer '{layer_name}'. Shape: {kernels_np.shape}")
                continue

            if kernels_np.ndim == 2:
                print(f"Skipping visualization for FC-like kernel '{layer_name}' (shape {kernels_np.shape}) in '{model_name}'. Visualization is for Conv kernels.")
                continue
            elif kernels_np.ndim == 3: # (H, W, Cout) or (Cin, H, W) - assume (H, W, Cout) for typical viz
                num_out_features = kernels_np.shape[2]
                kernels_to_plot_slices = [kernels_np[:, :, k] for k in range(min(num_kernels_to_show, num_out_features))]
            elif kernels_np.ndim == 4: # (KH, KW, Cin, Cout)
                num_out_features = kernels_np.shape[3]
                # Plot first input channel for each output feature map
                kernels_to_plot_slices = [kernels_np[:, :, 0, k] for k in range(min(num_kernels_to_show, num_out_features))]
            else:
                 print(f"Unexpected kernel ndim {kernels_np.ndim} for '{model_name}' layer '{layer_name}'.")
                 continue

            num_actual_kernels_shown = len(kernels_to_plot_slices)
            if num_actual_kernels_shown == 0:
                print(f"No kernel slices to plot for '{model_name}' layer '{layer_name}'.")
                continue

            cols_k = int(np.ceil(np.sqrt(num_actual_kernels_shown)))
            rows_k = int(np.ceil(num_actual_kernels_shown / cols_k))

            fig_k, axes_k = plt.subplots(rows_k, cols_k, figsize=(cols_k * 2.5, rows_k * 2.5), squeeze=False)
            fig_k.suptitle(f'Model: {model_name} - Layer: {layer_name} Kernels (First {num_actual_kernels_shown})', fontsize=14)
            axes_k_flat = axes_k.flatten()

            for k_idx in range(num_actual_kernels_shown):
                ax = axes_k_flat[k_idx]
                kernel_slice = kernels_to_plot_slices[k_idx]
                if kernel_slice.size > 0:
                    min_val, max_val = np.min(kernel_slice), np.max(kernel_slice)
                    norm_slice = (kernel_slice - min_val) / (max_val - min_val + 1e-5) if (max_val - min_val) > 1e-5 else kernel_slice
                    ax.imshow(norm_slice, cmap='viridis', aspect='auto')
                ax.set_title(f'Kernel {k_idx+1}'); ax.axis('off')            
                for k_j in range(num_actual_kernels_shown, len(axes_k_flat)):
                    fig_k.delaxes(axes_k_flat[k_j])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the plot
            if save_dir is None:
                save_dir = create_viz_directory()
            
            kernels_dir = os.path.join(save_dir, "kernels")
            os.makedirs(kernels_dir, exist_ok=True)
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
            filename = f"kernels_{safe_model_name}_{safe_layer_name}.png"
            filepath = os.path.join(kernels_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"🖼️ Kernels for {model_name} saved to: {filepath}")
        except Exception as e_plot:
            print(f"Error during kernel plotting for model '{model_name}', layer '{layer_name}': {e_plot}")

    print("\n🖼️ Kernel visualization attempt for all models complete.")

def get_activation_maps(model, layer_name, input_sample, training=False):
    try:
        if not callable(model):
            print(f"Error: Model object {type(model)} is not callable.")
            return None
        # Model __call__ should handle return_activations_for_layer
        activations = model(input_sample, training=training, return_activations_for_layer=layer_name)
        return activations
    except AttributeError as e_attr:
        print(f"AttributeError in {model.__class__.__name__} when getting activations for '{layer_name}': {e_attr}.")
        return None
    except Exception as e_general:
        print(f"An unexpected error occurred in {model.__class__.__name__} for layer '{layer_name}': {e_general}")
        return None

def activation_map_visualization_all(models_list: list[nnx.Module], model_names_list: list[str],
                                     layer_name_map: dict[str, str], test_ds_iter, num_maps_to_show=16, save_dir: str = None):
    num_models = len(models_list)
    if num_models == 0 or len(model_names_list) != num_models or len(layer_name_map) != num_models:
        print("Error: Invalid input for activation_map_visualization_all.")
        return

    print(f"\n🗺️ VISUALIZING ACTIVATION MAPS FOR ALL {num_models} MODELS"); print("=" * 60)

    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except StopIteration:
        print("Error: Test dataset iterator is exhausted. Cannot get sample batch for activation maps.")
        return
    except Exception as e:
        print(f"Error getting sample batch for activation maps: {e}")
        return

    if 'image' not in sample_batch or sample_batch['image'].shape[0] == 0:
        print("Error: Sample batch does not contain 'image' or is empty.")
        return
    input_sample_np = sample_batch['image'][0:1] # Take first image, keep batch dim
    input_sample_jax = jnp.asarray(input_sample_np)

    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        layer_name = layer_name_map.get(model_name)

        if not layer_name:
            print(f"Warning: Layer name for activation map visualization not specified for model '{model_name}'. Skipping.")
            continue

        print(f"\n🗺️ Visualizing activation maps from layer '{layer_name}' for model '{model_name}'...")
        activations = get_activation_maps(model, layer_name, input_sample_jax, training=False)

        if activations is None:
            print(f"Could not retrieve activation maps for layer '{layer_name}' in model '{model_name}'.")
            continue

        try:
            activations_np = np.array(activations)
            if activations_np.ndim < 3: # Expect (H, W, C) or (B, H, W, C)
                print(f"Unexpected activation shape for {model_name}, layer '{layer_name}': {activations_np.shape}. Expected at least 3 dims.")
                continue

            if activations_np.ndim == 4 and activations_np.shape[0] == 1:
                activations_np_plot = activations_np[0]
            elif activations_np.ndim == 3:
                activations_np_plot = activations_np
            else:
                print(f"Cannot properly process activation shape {activations_np.shape} for plotting for model {model_name}, layer '{layer_name}'.")
                continue

            num_channels = activations_np_plot.shape[-1]
            actual_maps_to_show = min(num_maps_to_show, num_channels)

            if actual_maps_to_show == 0:
                print(f"No activation maps to show (num_channels={num_channels}) for {model_name}, layer '{layer_name}'.")
                continue

            cols_am = int(np.ceil(np.sqrt(actual_maps_to_show)))
            rows_am = int(np.ceil(actual_maps_to_show / cols_am))

            fig_am, axes_am = plt.subplots(rows_am, cols_am, figsize=(cols_am * 2, rows_am * 2), squeeze=False)
            fig_am.suptitle(f'Model: {model_name} - Layer: {layer_name} Activations (First {actual_maps_to_show})', fontsize=14)
            axes_am_flat = axes_am.flatten()

            for m_idx in range(actual_maps_to_show):
                ax = axes_am_flat[m_idx]
                act_map_slice = activations_np_plot[:, :, m_idx]
                if act_map_slice.size > 0:
                    ax.imshow(act_map_slice, cmap='viridis')
                ax.set_title(f'Map {m_idx+1}'); ax.axis('off')            
                for m_j in range(actual_maps_to_show, len(axes_am_flat)):
                    fig_am.delaxes(axes_am_flat[m_j])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the plot
            if save_dir is None:
                save_dir = create_viz_directory()
            
            activations_dir = os.path.join(save_dir, "activation_maps")
            os.makedirs(activations_dir, exist_ok=True)
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
            filename = f"activations_{safe_model_name}_{safe_layer_name}.png"
            filepath = os.path.join(activations_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"🗺️ Activation maps for {model_name} saved to: {filepath}")
        except Exception as e_plot:
            print(f"Error during activation map plotting for model '{model_name}', layer '{layer_name}': {e_plot}")

    print("\n🗺️ Activation map visualization attempt for all models complete.")


def saliency_map_analysis_all(models_list: list[nnx.Module], model_names_list: list[str], test_ds_iter, class_names: list[str], save_dir: str = None):
    num_models = len(models_list)
    if num_models == 0 or len(model_names_list) != num_models:
        print("Error: Invalid input for saliency_map_analysis_all.")
        return
    if not class_names:
        print("Error: class_names list is empty for saliency_map_analysis_all.")
        return

    print(f"\n🔥 SALIENCY MAP ANALYSIS (ALL {num_models} MODELS) for {len(class_names)} classes"); print("=" * 70)

    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except StopIteration:
        print("ERROR: Test dataset iterator is exhausted for saliency maps. Consider re-creating it.")
        return
    except Exception as e:
        print(f"Error: Could not get sample batch for saliency maps: {e}")
        return

    if 'image' not in sample_batch or sample_batch['image'].shape[0] == 0 or \
       'label' not in sample_batch or sample_batch['label'].shape[0] == 0:
        print("Error: Sample batch is missing 'image' or 'label', or they are empty.")
        return

    sample_image_np = sample_batch['image'][0:1]
    sample_image_jax = jnp.asarray(sample_image_np)
    sample_label_true_idx = int(sample_batch['label'][0])

    if not (0 <= sample_label_true_idx < len(class_names)):
        print(f"Error: True label index {sample_label_true_idx} is out of bounds for class_names list (len {len(class_names)}).")
        return
    true_class_name = class_names[sample_label_true_idx]

    @partial(jax.jit, static_argnums=(0,2)) # model_obj and class_idx_saliency are static
    def get_saliency_map_for_model(model_obj, image_input, class_idx_saliency=None):
        def model_output_for_grad(img_grad):
            logits_saliency = model_obj(img_grad, training=False)
            num_model_classes_saliency = logits_saliency.shape[-1]

            target_idx = class_idx_saliency
            if class_idx_saliency is None:
                target_idx = jnp.argmax(logits_saliency[0])
            elif not (0 <= class_idx_saliency < num_model_classes_saliency):
                print(f"Warning: Saliency class_idx {class_idx_saliency} out of bounds for model ({num_model_classes_saliency} classes). Using argmax.")
                target_idx = jnp.argmax(logits_saliency[0])
            return logits_saliency[0, target_idx]

        grads = jax.grad(model_output_for_grad)(image_input)
        saliency = jnp.max(jnp.abs(grads[0]), axis=-1)
        return saliency

    print(f"Analyzing saliency for a sample image. True Class: {true_class_name} (Index: {sample_label_true_idx})")

    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        print(f"\nProcessing saliency for model: {model_name}...")

        try:
            logits_model_sample = model(sample_image_jax, training=False)
            num_model_classes = logits_model_sample.shape[-1]
            if not (0 <= sample_label_true_idx < num_model_classes):
                 print(f"Warning: True label index {sample_label_true_idx} is out of bounds for {model_name}'s output ({num_model_classes} classes).")

            model_predicted_class_idx = int(jnp.argmax(logits_model_sample, axis=1)[0])
            if not (0 <= model_predicted_class_idx < len(class_names)):
                print(f"Error: Predicted label index {model_predicted_class_idx} by {model_name} is out of bounds for class_names list.")
                model_predicted_class_name = "ErrorIdx"
            else:
                model_predicted_class_name = class_names[model_predicted_class_idx]
            print(f"  {model_name} predicted class: {model_predicted_class_name} (Index: {model_predicted_class_idx})")

            saliency_true_class = get_saliency_map_for_model(model, sample_image_jax, class_idx_saliency=sample_label_true_idx)
            saliency_pred_class_model = get_saliency_map_for_model(model, sample_image_jax, class_idx_saliency=model_predicted_class_idx)

            fig_sal, axes_sal = plt.subplots(1, 3, figsize=(15, 5))
            fig_sal.suptitle(f'Saliency Maps - Model: {model_name}\nTrue Class: {true_class_name} | Predicted: {model_predicted_class_name}', fontsize=14)

            img_display_np = np.array(sample_image_np[0])
            min_img, max_img = np.min(img_display_np), np.max(img_display_np)
            img_display_norm = (img_display_np - min_img) / (max_img - min_img + 1e-5) if (max_img - min_img) > 1e-5 else img_display_np

            axes_sal[0].imshow(img_display_norm)
            axes_sal[0].set_title("Original Image")
            axes_sal[0].axis('off')

            im_true = axes_sal[1].imshow(np.array(saliency_true_class), cmap='hot')
            axes_sal[1].set_title(f"Saliency for True ({true_class_name})")
            axes_sal[1].axis('off')
            fig_sal.colorbar(im_true, ax=axes_sal[1], fraction=0.046, pad=0.04)            
            im_pred = axes_sal[2].imshow(np.array(saliency_pred_class_model), cmap='hot')
            axes_sal[2].set_title(f"Saliency for Pred ({model_predicted_class_name})")
            axes_sal[2].axis('off')
            fig_sal.colorbar(im_pred, ax=axes_sal[2], fraction=0.046, pad=0.04)

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            
            # Save the plot
            if save_dir is None:
                save_dir = create_viz_directory()
            
            saliency_dir = os.path.join(save_dir, "saliency_maps")
            os.makedirs(saliency_dir, exist_ok=True)
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            filename = f"saliency_{safe_model_name}.png"
            filepath = os.path.join(saliency_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"🔥 Saliency maps for {model_name} saved to: {filepath}")

        except Exception as e_model_saliency:
            print(f"Error during saliency analysis for model '{model_name}': {e_model_saliency}")

    print("\n🔥 Saliency map analysis attempt for all models complete.")
