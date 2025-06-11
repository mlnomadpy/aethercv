import typing as tp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
import scipy.ndimage
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

Array = jax.Array

# ===== ADVANCED GRADIENT-BASED EXPLAINABILITY =====

def guided_backprop_all(models_list: list[nnx.Module], model_names_list: list[str], 
                       test_ds_iter, class_names: list[str]):
    """
    Guided Backpropagation - modifies gradients during backprop to only show positive contributions
    """
    print(f"\n🎯 GUIDED BACKPROPAGATION ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except:
        print("Error: Could not get sample batch for guided backprop")
        return
    
    sample_image = jnp.asarray(sample_batch['image'][0:1])
    true_label = int(sample_batch['label'][0])
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        
        @jax.jit
        def guided_gradients(image_input, target_class):
            def model_pred_fn(img):
                logits = model(img, training=False)
                return logits[0, target_class]
            
            # Standard gradients
            grads = jax.grad(model_pred_fn)(image_input)
            
            # Apply ReLU to gradients (guided backprop principle)
            guided_grads = jnp.maximum(grads, 0)
            return guided_grads
        
        guided_grad_map = guided_gradients(sample_image, true_label)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Guided Backpropagation - {model_name}', fontsize=14)
        
        # Original image
        axes[0].imshow(np.array(sample_image[0]))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Raw gradients magnitude
        grad_magnitude = jnp.sqrt(jnp.sum(guided_grad_map[0]**2, axis=-1))
        axes[1].imshow(grad_magnitude, cmap='hot')
        axes[1].set_title('Guided Gradients')
        axes[1].axis('off')
        
        # Overlaid on original
        overlay = np.array(sample_image[0]) * 0.7 + np.stack([grad_magnitude]*3, axis=-1) * 0.3
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


def integrated_gradients_all(models_list: list[nnx.Module], model_names_list: list[str],
                           test_ds_iter, class_names: list[str], num_steps: int = 50):
    """
    Integrated Gradients - integrates gradients along path from baseline to input
    """
    print(f"\n🔗 INTEGRATED GRADIENTS ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except:
        print("Error: Could not get sample batch for integrated gradients")
        return
    
    sample_image = jnp.asarray(sample_batch['image'][0:1])
    true_label = int(sample_batch['label'][0])
    
    # Baseline (typically zeros or blurred image)
    baseline = jnp.zeros_like(sample_image)
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        
        @jax.jit
        def compute_integrated_gradients(image_input, baseline_input, target_class, num_steps):
            # Create interpolated inputs
            alphas = jnp.linspace(0, 1, num_steps)
            
            def single_step_grad(alpha):
                interpolated = baseline_input + alpha * (image_input - baseline_input)
                
                def model_pred_fn(img):
                    logits = model(img, training=False)
                    return logits[0, target_class]
                
                return jax.grad(model_pred_fn)(interpolated)
            
            # Vectorized computation of gradients at all interpolation points
            gradients = jax.vmap(single_step_grad)(alphas)
            
            # Integrate using trapezoidal rule
            integrated_grads = jnp.mean(gradients, axis=0) * (image_input - baseline_input)
            return integrated_grads
        
        integrated_grad_map = compute_integrated_gradients(sample_image, baseline, true_label, num_steps)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Integrated Gradients - {model_name}', fontsize=14)
        
        axes[0].imshow(np.array(sample_image[0]))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        grad_magnitude = jnp.sqrt(jnp.sum(integrated_grad_map[0]**2, axis=-1))
        im = axes[1].imshow(grad_magnitude, cmap='RdBu_r')
        axes[1].set_title('Integrated Gradients')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Attribution overlay
        overlay = np.array(sample_image[0]) * 0.7 + np.stack([grad_magnitude]*3, axis=-1) * 0.3
        axes[2].imshow(overlay)
        axes[2].set_title('Attribution Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


# ===== FEATURE REPRESENTATION ANALYSIS =====

def feature_space_visualization_all(models_list: list[nnx.Module], model_names_list: list[str],
                                  test_ds_iter, class_names: list[str], 
                                  layer_names: dict[str, str], max_samples: int = 1000):
    """
    Visualize feature representations using t-SNE and PCA
    """
    print(f"\n🗺️ FEATURE SPACE VISUALIZATION ({len(models_list)} MODELS)")
    print("=" * 70)
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        layer_name = layer_names.get(model_name)
        
        if not layer_name:
            print(f"No layer specified for {model_name}, skipping...")
            continue
        
        # Collect features and labels
        features_list = []
        labels_list = []
        sample_count = 0
        
        for batch in test_ds_iter.as_numpy_iterator():
            if sample_count >= max_samples:
                break
                
            images = jnp.asarray(batch['image'])
            labels = batch['label']
            
            # Extract features from specified layer
            batch_features = model(images, training=False, return_activations_for_layer=layer_name)
            
            if batch_features is not None:
                # Flatten features for each sample
                batch_features_flat = batch_features.reshape(batch_features.shape[0], -1)
                features_list.append(np.array(batch_features_flat))
                labels_list.append(labels)
                sample_count += len(labels)
        
        if not features_list:
            print(f"No features extracted for {model_name}")
            continue
        
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        
        # Apply PCA first to reduce dimensionality
        pca = PCA(n_components=min(50, features.shape[1]))
        features_pca = pca.fit_transform(features)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(features_pca)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Feature Space Visualization - {model_name} (Layer: {layer_name})', fontsize=14)
        
        # PCA visualization (first 2 components)
        scatter1 = ax1.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, 
                              cmap='tab10', alpha=0.6, s=20)
        ax1.set_title(f'PCA (Explained Variance: {pca.explained_variance_ratio_[:2].sum():.2f})')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        
        # t-SNE visualization
        scatter2 = ax2.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, 
                              cmap='tab10', alpha=0.6, s=20)
        ax2.set_title('t-SNE')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        
        # Add colorbar
        plt.colorbar(scatter2, ax=ax2, label='Class')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate clustering metrics
        if len(np.unique(labels)) > 1:
            silhouette_pca = silhouette_score(features_pca[:, :2], labels)
            silhouette_tsne = silhouette_score(features_tsne, labels)
            print(f"  {model_name} - Silhouette Score PCA: {silhouette_pca:.3f}, t-SNE: {silhouette_tsne:.3f}")


def layer_wise_activation_analysis_all(models_list: list[nnx.Module], model_names_list: list[str],
                                     test_ds_iter, sample_layers: dict[str, list[str]]):
    """
    Analyze activation patterns across different layers
    """
    print(f"\n📊 LAYER-WISE ACTIVATION ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except:
        print("Error: Could not get sample batch")
        return
    
    sample_image = jnp.asarray(sample_batch['image'][0:1])
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        layers = sample_layers.get(model_name, [])
        
        if not layers:
            print(f"No layers specified for {model_name}")
            continue
        
        activation_stats = {}
        
        for layer_name in layers:
            try:
                activations = model(sample_image, training=False, 
                                  return_activations_for_layer=layer_name)
                
                if activations is not None:
                    act_array = np.array(activations)
                    activation_stats[layer_name] = {
                        'mean': np.mean(act_array),
                        'std': np.std(act_array),
                        'sparsity': np.mean(act_array == 0),  # Fraction of zeros
                        'max': np.max(act_array),
                        'min': np.min(act_array),
                        'shape': act_array.shape
                    }
            except Exception as e:
                print(f"Error getting activations for {layer_name}: {e}")
        
        if activation_stats:
            # Create visualization
            metrics = ['mean', 'std', 'sparsity']
            fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
            fig.suptitle(f'Layer-wise Activation Statistics - {model_name}', fontsize=14)
            
            layer_names_plot = list(activation_stats.keys())
            
            for j, metric in enumerate(metrics):
                values = [activation_stats[layer][metric] for layer in layer_names_plot]
                axes[j].bar(range(len(layer_names_plot)), values)
                axes[j].set_title(f'{metric.capitalize()}')
                axes[j].set_xticks(range(len(layer_names_plot)))
                axes[j].set_xticklabels(layer_names_plot, rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Print detailed stats
            print(f"\n{model_name} Activation Statistics:")
            for layer_name, stats in activation_stats.items():
                print(f"  {layer_name}: shape={stats['shape']}, mean={stats['mean']:.4f}, "
                      f"std={stats['std']:.4f}, sparsity={stats['sparsity']:.3f}")


# ===== ATTENTION AND INFLUENCE ANALYSIS =====

def grad_cam_analysis_all(models_list: list[nnx.Module], model_names_list: list[str],
                         test_ds_iter, class_names: list[str], 
                         target_layers: dict[str, str]):
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping)
    """
    print(f"\n🎯 GRAD-CAM ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except:
        print("Error: Could not get sample batch for Grad-CAM")
        return
    
    sample_image = jnp.asarray(sample_batch['image'][0:1])
    true_label = int(sample_batch['label'][0])
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        target_layer = target_layers.get(model_name)
        
        if not target_layer:
            print(f"No target layer specified for {model_name}")
            continue
        
        @jax.jit
        def compute_grad_cam(image_input, target_class):
            def get_features_and_gradients(img):
                # Get activations from target layer
                features = model(img, training=False, return_activations_for_layer=target_layer)
                
                # Get model output
                logits = model(img, training=False)
                target_score = logits[0, target_class]
                
                return features, target_score
            
            # Get features and compute gradients
            features, target_score = get_features_and_gradients(image_input)
            grad_fn = jax.grad(lambda img: get_features_and_gradients(img)[1])
            gradients = grad_fn(image_input)
            
            # For Grad-CAM, we need gradients w.r.t. feature maps
            # This is a simplified version - in practice, you'd need to modify the model
            # to return intermediate gradients
            
            return features, gradients
        
        try:
            features, gradients = compute_grad_cam(sample_image, true_label)
            
            if features is not None:
                # Create heatmap visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'Grad-CAM Analysis - {model_name}', fontsize=14)
                
                axes[0].imshow(np.array(sample_image[0]))
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Simplified heatmap (normally would be weighted by gradients)
                if len(features.shape) == 4:  # (B, H, W, C)
                    heatmap = np.mean(np.array(features[0]), axis=-1)
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                    
                    axes[1].imshow(heatmap, cmap='jet', alpha=0.8)
                    axes[1].set_title(f'Activation Heatmap ({target_layer})')
                    axes[1].axis('off')
                    
                    # Overlay on original
                    heatmap_resized = scipy.ndimage.zoom(heatmap, 
                                                       (sample_image.shape[1]/heatmap.shape[0],
                                                        sample_image.shape[2]/heatmap.shape[1]), 
                                                       order=1)
                    overlay = np.array(sample_image[0]) * 0.6
                    overlay[:,:,0] += heatmap_resized * 0.4  # Add red channel
                    overlay = np.clip(overlay, 0, 1)
                    
                    axes[2].imshow(overlay)
                    axes[2].set_title('Overlay')
                    axes[2].axis('off')
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Error in Grad-CAM for {model_name}: {e}")


# ===== COMPARATIVE ANALYSIS METHODS =====

def model_decision_boundary_comparison(models_list: list[nnx.Module], model_names_list: list[str],
                                     test_ds_iter, class_names: list[str], 
                                     num_samples: int = 500):
    """
    Compare how different models make decisions on the same inputs
    """
    print(f"\n⚖️ MODEL DECISION BOUNDARY COMPARISON ({len(models_list)} MODELS)")
    print("=" * 70)
    
    # Collect predictions from all models
    all_predictions = {name: [] for name in model_names_list}
    all_confidences = {name: [] for name in model_names_list}
    true_labels = []
    sample_count = 0
    
    for batch in test_ds_iter.as_numpy_iterator():
        if sample_count >= num_samples:
            break
        
        images = jnp.asarray(batch['image'])
        labels = batch['label']
        
        for i, model_name in enumerate(model_names_list):
            model = models_list[i]
            logits = model(images, training=False)
            predictions = jnp.argmax(logits, axis=1)
            confidences = jnp.max(jax.nn.softmax(logits), axis=1)
            
            all_predictions[model_name].extend(np.array(predictions))
            all_confidences[model_name].extend(np.array(confidences))
        
        true_labels.extend(labels)
        sample_count += len(labels)
    
    # Convert to arrays
    for model_name in model_names_list:
        all_predictions[model_name] = np.array(all_predictions[model_name])
        all_confidences[model_name] = np.array(all_confidences[model_name])
    true_labels = np.array(true_labels)
    
    # Analysis and visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Decision Boundary Comparison', fontsize=16)
    
    # 1. Agreement matrix between models
    if len(model_names_list) >= 2:
        agreement_matrix = np.zeros((len(model_names_list), len(model_names_list)))
        
        for i in range(len(model_names_list)):
            for j in range(len(model_names_list)):
                if i != j:
                    agreements = (all_predictions[model_names_list[i]] == 
                                all_predictions[model_names_list[j]]).mean()
                    agreement_matrix[i, j] = agreements
                else:
                    agreement_matrix[i, j] = 1.0
        
        im1 = axes[0, 0].imshow(agreement_matrix, cmap='Blues', vmin=0, vmax=1)
        axes[0, 0].set_title('Model Agreement Matrix')
        axes[0, 0].set_xticks(range(len(model_names_list)))
        axes[0, 0].set_yticks(range(len(model_names_list)))
        axes[0, 0].set_xticklabels(model_names_list, rotation=45)
        axes[0, 0].set_yticklabels(model_names_list)
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Add text annotations
        for i in range(len(model_names_list)):
            for j in range(len(model_names_list)):
                axes[0, 0].text(j, i, f'{agreement_matrix[i, j]:.3f}', 
                               ha='center', va='center')
    
    # 2. Confidence distributions
    axes[0, 1].set_title('Confidence Distributions')
    for model_name in model_names_list:
        axes[0, 1].hist(all_confidences[model_name], alpha=0.7, label=model_name, bins=30)
    axes[0, 1].set_xlabel('Prediction Confidence')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Disagreement analysis
    disagreement_cases = np.zeros(len(true_labels), dtype=bool)
    if len(model_names_list) >= 2:
        for i in range(len(model_names_list)):
            for j in range(i+1, len(model_names_list)):
                disagreements = (all_predictions[model_names_list[i]] != 
                               all_predictions[model_names_list[j]])
                disagreement_cases |= disagreements
    
    correct_cases = np.array([all_predictions[model_names_list[0]] == true_labels])
    for model_name in model_names_list[1:]:
        correct_cases = np.vstack([correct_cases, 
                                 all_predictions[model_name] == true_labels])
    
    disagreement_types = ['All Agree & Correct', 'All Agree & Wrong', 
                         'Disagree & Majority Correct', 'Disagree & Majority Wrong']
    
    # Calculate disagreement statistics
    all_agree = ~disagreement_cases
    all_correct = np.all(correct_cases, axis=0)
    majority_correct = np.sum(correct_cases, axis=0) > len(model_names_list) // 2
    
    counts = [
        np.sum(all_agree & all_correct),
        np.sum(all_agree & ~all_correct),
        np.sum(disagreement_cases & majority_correct),
        np.sum(disagreement_cases & ~majority_correct)
    ]
    
    axes[1, 0].pie(counts, labels=disagreement_types, autopct='%1.1f%%')
    axes[1, 0].set_title('Model Agreement Analysis')
    
    # 4. Per-class accuracy comparison
    class_accuracies = {}
    unique_classes = np.unique(true_labels)
    
    for model_name in model_names_list:
        class_acc = []
        for class_idx in unique_classes:
            class_mask = true_labels == class_idx
            if np.sum(class_mask) > 0:
                acc = np.mean(all_predictions[model_name][class_mask] == class_idx)
                class_acc.append(acc)
            else:
                class_acc.append(0)
        class_accuracies[model_name] = class_acc
    
    x_pos = np.arange(len(unique_classes))
    width = 0.8 / len(model_names_list)
    
    for i, model_name in enumerate(model_names_list):
        axes[1, 1].bar(x_pos + i * width, class_accuracies[model_name], 
                      width, label=model_name, alpha=0.8)
    
    axes[1, 1].set_title('Per-Class Accuracy Comparison')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_xticks(x_pos + width * (len(model_names_list) - 1) / 2)
    axes[1, 1].set_xticklabels([class_names[i] if i < len(class_names) else f'Class {i}' 
                               for i in unique_classes], rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nDecision Boundary Analysis Summary:")
    print("-" * 50)
    for i, model_name in enumerate(model_names_list):
        accuracy = np.mean(all_predictions[model_name] == true_labels)
        avg_confidence = np.mean(all_confidences[model_name])
        print(f"{model_name}: Accuracy={accuracy:.3f}, Avg Confidence={avg_confidence:.3f}")
    
    if len(model_names_list) >= 2:
        overall_agreement = np.mean(~disagreement_cases)
        print(f"\nOverall Model Agreement: {overall_agreement:.3f}")


# ===== UNCERTAINTY AND ROBUSTNESS ANALYSIS =====

def prediction_uncertainty_analysis_all(models_list: list[nnx.Module], model_names_list: list[str],
                                       test_ds_iter, class_names: list[str],
                                       num_samples: int = 200):
    """
    Analyze prediction uncertainty using entropy and confidence measures
    """
    print(f"\n🎲 PREDICTION UNCERTAINTY ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    model_uncertainties = {name: {'entropy': [], 'max_prob': [], 'correct': []} 
                          for name in model_names_list}
    
    sample_count = 0
    for batch in test_ds_iter.as_numpy_iterator():
        if sample_count >= num_samples:
            break
        
        images = jnp.asarray(batch['image'])
        labels = batch['label']
        
        for i, model_name in enumerate(model_names_list):
            model = models_list[i]
            logits = model(images, training=False)
            probs = jax.nn.softmax(logits)
            
            # Calculate entropy (uncertainty measure)
            entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=1)
            max_prob = jnp.max(probs, axis=1)
            predictions = jnp.argmax(logits, axis=1)
            correct = predictions == labels
            
            model_uncertainties[model_name]['entropy'].extend(np.array(entropy))
            model_uncertainties[model_name]['max_prob'].extend(np.array(max_prob))
            model_uncertainties[model_name]['correct'].extend(np.array(correct))
        
        sample_count += len(labels)
    
    # Convert to arrays
    for model_name in model_names_list:
        for key in model_uncertainties[model_name]:
            model_uncertainties[model_name][key] = np.array(model_uncertainties[model_name][key])
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Prediction Uncertainty Analysis', fontsize=16)
    
    # 1. Entropy distributions
    for model_name in model_names_list:
        axes[0, 0].hist(model_uncertainties[model_name]['entropy'], 
                       alpha=0.7, label=model_name, bins=30)
    axes[0, 0].set_xlabel('Prediction Entropy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Entropy Distributions')
    axes[0, 0].legend()
    
    # 2. Confidence vs Accuracy
    for model_name in model_names_list:
        confidences = model_uncertainties[model_name]['max_prob']
        correct = model_uncertainties[model_name]['correct']
        
        # Bin by confidence and calculate accuracy in each bin
        bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_centers = []
        
        for i in range(len(bins)-1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(correct[mask])
                bin_accuracies.append(bin_acc)
                bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        axes[0, 1].plot(bin_centers, bin_accuracies, 'o-', label=model_name, linewidth=2)
    
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Confidence vs Accuracy (Calibration)')
    axes[0, 1].legend()
    
    # 3. Uncertainty vs Correctness
    for model_name in model_names_list:
        entropy = model_uncertainties[model_name]['entropy']
        correct = model_uncertainties[model_name]['correct']
        
        correct_entropy = entropy[correct]
        incorrect_entropy = entropy[~correct]
        
        axes[1, 0].hist(correct_entropy, alpha=0.5, label=f'{model_name} (Correct)', 
                       bins=20, density=True)
        axes[1, 0].hist(incorrect_entropy, alpha=0.5, label=f'{model_name} (Incorrect)', 
                       bins=20, density=True)
    
    axes[1, 0].set_xlabel('Entropy')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Entropy Distribution: Correct vs Incorrect')
    axes[1, 0].legend()
    
    # 4. Model uncertainty comparison
    uncertainty_metrics = []
    model_names_plot = []
    
    for model_name in model_names_list:
        avg_entropy = np.mean(model_uncertainties[model_name]['entropy'])
        avg_confidence = np.mean(model_uncertainties[model_name]['max_prob'])
        accuracy = np.mean(model_uncertainties[model_name]['correct'])
        
        uncertainty_metrics.append([avg_entropy, 1-avg_confidence, 1-accuracy])
        model_names_plot.append(model_name)
    
    uncertainty_metrics = np.array(uncertainty_metrics)
    metric_names = ['Avg Entropy', 'Avg Uncertainty', 'Error Rate']
    
    x_pos = np.arange(len(model_names_plot))
    width = 0.25
    
    for i, metric_name in enumerate(metric_names):
        axes[1, 1].bar(x_pos + i * width, uncertainty_metrics[:, i], 
                      width, label=metric_name, alpha=0.8)
    
    axes[1, 1].set_title('Uncertainty Metrics Comparison')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x_pos + width)
    axes[1, 1].set_xticklabels(model_names_plot)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\nUncertainty Analysis Summary:")
    print("-" * 50)
    for model_name in model_names_list:
        avg_entropy = np.mean(model_uncertainties[model_name]['entropy'])
        avg_confidence = np.mean(model_uncertainties[model_name]['max_prob'])
        accuracy = np.mean(model_uncertainties[model_name]['correct'])
        
        print(f"{model_name}:")
        print(f"  Average Entropy: {avg_entropy:.4f}")
        print(f"  Average Confidence: {avg_confidence:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print()
