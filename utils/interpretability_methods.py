import typing as tp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
import networkx as nx
import os
from datetime import datetime

Array = jax.Array

# Create directory for saving visualizations
def create_viz_directory():
    """Create a directory for saving visualizations with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = f"visualizations_{timestamp}"
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir

# ===== FILTER AND FEATURE ANALYSIS =====

def filter_similarity_analysis_all(models_list: list[nnx.Module], model_names_list: list[str],
                                  layer_names: dict[str, str], save_dir: str = None):
    """
    Analyze similarity patterns between filters in convolutional layers
    """
    print(f"\n🔍 FILTER SIMILARITY ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    if save_dir is None:
        save_dir = create_viz_directory()
    
    filter_dir = os.path.join(save_dir, "filter_analysis")
    os.makedirs(filter_dir, exist_ok=True)
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        layer_name = layer_names.get(model_name)
        
        if not layer_name:
            print(f"No layer specified for {model_name}")
            continue
        
        # Extract weights from the specified layer
        try:
            # This is a simplified approach - you'd need to adapt based on your model structure
            layer_weights = None
            
            # Try to find the layer weights
            def find_layer_weights(module, target_name, current_path=""):
                nonlocal layer_weights
                
                if hasattr(module, '__dict__'):
                    for name, attr in module.__dict__.items():
                        full_path = f"{current_path}.{name}" if current_path else name
                        
                        if full_path == target_name or name == target_name:
                            if hasattr(attr, 'kernel') and isinstance(attr.kernel, jnp.ndarray):
                                layer_weights = attr.kernel
                                return
                            elif isinstance(attr, jnp.ndarray) and len(attr.shape) == 4:
                                layer_weights = attr
                                return
                        
                        if hasattr(attr, '__dict__'):
                            find_layer_weights(attr, target_name, full_path)
            
            find_layer_weights(model, layer_name)
            
            if layer_weights is None:
                print(f"Could not find weights for layer {layer_name} in {model_name}")
                continue
                
            weights = np.array(layer_weights)
            
            if len(weights.shape) != 4:
                print(f"Expected 4D weights (H, W, C_in, C_out), got {weights.shape}")
                continue
            
            H, W, C_in, C_out = weights.shape
            
            # Flatten each filter for similarity computation
            filters_flat = weights.reshape(-1, C_out).T  # (C_out, H*W*C_in)
            
            # Compute pairwise cosine similarity
            filter_norms = np.linalg.norm(filters_flat, axis=1, keepdims=True)
            normalized_filters = filters_flat / (filter_norms + 1e-8)
            similarity_matrix = np.dot(normalized_filters, normalized_filters.T)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Filter Analysis - {model_name} (Layer: {layer_name})', fontsize=16)
            
            # 1. Similarity matrix heatmap
            im1 = axes[0, 0].imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, 0].set_title('Filter Similarity Matrix')
            axes[0, 0].set_xlabel('Filter Index')
            axes[0, 0].set_ylabel('Filter Index')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # 2. Similarity distribution
            # Get upper triangle (excluding diagonal) for distribution
            triu_indices = np.triu_indices_from(similarity_matrix, k=1)
            similarities = similarity_matrix[triu_indices]
            
            axes[0, 1].hist(similarities, bins=50, alpha=0.7, density=True)
            axes[0, 1].set_xlabel('Cosine Similarity')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Filter Similarity Distribution')
            axes[0, 1].axvline(np.mean(similarities), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(similarities):.3f}')
            axes[0, 1].legend()
            
            # 3. Hierarchical clustering of filters
            if C_out <= 50:  # Only for manageable number of filters
                distance_matrix = 1 - similarity_matrix
                linkage_matrix = sch.linkage(distance_matrix, method='ward')
                
                dendro = sch.dendrogram(linkage_matrix, ax=axes[1, 0], 
                                      truncate_mode='lastp', p=min(20, C_out))
                axes[1, 0].set_title('Filter Hierarchical Clustering')
                axes[1, 0].set_xlabel('Filter Clusters')
                axes[1, 0].set_ylabel('Distance')
            else:
                axes[1, 0].text(0.5, 0.5, f'Too many filters ({C_out}) for clustering visualization', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Clustering Skipped')
            
            # 4. Filter redundancy analysis
            redundancy_threshold = 0.9
            high_similarity_pairs = np.where(similarity_matrix > redundancy_threshold)
            # Remove diagonal elements
            mask = high_similarity_pairs[0] != high_similarity_pairs[1]
            redundant_pairs = list(zip(high_similarity_pairs[0][mask], high_similarity_pairs[1][mask]))
            
            redundancy_ratio = len(redundant_pairs) / (C_out * (C_out - 1) / 2)
            
            # Visualize filter activation ranges
            filter_ranges = np.max(filters_flat, axis=1) - np.min(filters_flat, axis=1)
            axes[1, 1].hist(filter_ranges, bins=20, alpha=0.7)
            axes[1, 1].set_xlabel('Filter Value Range')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title(f'Filter Dynamic Range\n(Redundancy: {redundancy_ratio:.1%})')
            plt.tight_layout()
            
            # Save the plot
            safe_model_name = model_name.replace(" ", "_").replace("/", "_")
            safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
            filename = f"filter_similarity_{safe_model_name}_{safe_layer_name}.png"
            filepath = os.path.join(filter_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved filter analysis plot: {filepath}")
            
            print(f"  {model_name} - Layer {layer_name}:")
            print(f"    Filters: {C_out}, Shape: {H}x{W}x{C_in}")
            print(f"    Mean similarity: {np.mean(similarities):.3f}")
            print(f"    Redundant pairs (>{redundancy_threshold}): {len(redundant_pairs)}")
            print(f"    Redundancy ratio: {redundancy_ratio:.1%}")
            
        except Exception as e:
            print(f"Error analyzing filters for {model_name}: {e}")


def feature_attribution_analysis_all(models_list: list[nnx.Module], model_names_list: list[str],
                                    test_ds_iter, class_names: list[str], save_dir: str = None):
    """
    Advanced feature attribution using multiple methods
    """
    print(f"\n🎯 FEATURE ATTRIBUTION ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    if save_dir is None:
        save_dir = create_viz_directory()
    
    attribution_dir = os.path.join(save_dir, "feature_attribution")
    os.makedirs(attribution_dir, exist_ok=True)
    
    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except:
        print("Error: Could not get sample batch")
        return
    
    sample_image = jnp.asarray(sample_batch['image'][0:1])
    true_label = int(sample_batch['label'][0])
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        
        # 1. Input * Gradient
        @jax.jit
        def input_x_gradient_attribution(image_input, target_class):
            def model_pred_fn(img):
                logits = model(img, training=False)
                return logits[0, target_class]
            
            gradients = jax.grad(model_pred_fn)(image_input)
            attribution = image_input * gradients
            return attribution
        
        # 2. SmoothGrad - average gradients over noisy inputs
        @jax.jit
        def smoothgrad_attribution(image_input, target_class, num_samples=50, noise_level=0.1):
            key = jax.random.PRNGKey(42)
            
            def single_noisy_grad(subkey):
                noise = jax.random.normal(subkey, image_input.shape) * noise_level
                noisy_input = image_input + noise
                
                def model_pred_fn(img):
                    logits = model(img, training=False)
                    return logits[0, target_class]
                
                return jax.grad(model_pred_fn)(noisy_input)
            
            keys = jax.random.split(key, num_samples)
            gradients = jax.vmap(single_noisy_grad)(keys)
            smooth_gradients = jnp.mean(gradients, axis=0)
            
            return smooth_gradients
        
        # 3. Gradient SHAP approximation
        @jax.jit
        def gradient_shap_attribution(image_input, target_class, num_baselines=10):
            key = jax.random.PRNGKey(42)
            
            def single_baseline_grad(subkey):
                # Random baseline
                baseline = jax.random.uniform(subkey, image_input.shape) * 0.1
                
                def model_pred_fn(img):
                    logits = model(img, training=False)
                    return logits[0, target_class]
                
                # Gradient at interpolated point
                alpha = 0.5  # Midpoint
                interpolated = baseline + alpha * (image_input - baseline)
                grad_interp = jax.grad(model_pred_fn)(interpolated)
                
                # SHAP-like attribution
                attribution = (image_input - baseline) * grad_interp
                return attribution
            
            keys = jax.random.split(key, num_baselines)
            attributions = jax.vmap(single_baseline_grad)(keys)
            shap_attribution = jnp.mean(attributions, axis=0)
            
            return shap_attribution
        
        # Compute all attributions
        input_grad_attr = input_x_gradient_attribution(sample_image, true_label)
        smooth_grad_attr = smoothgrad_attribution(sample_image, true_label)
        shap_attr = gradient_shap_attribution(sample_image, true_label)
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Feature Attribution Analysis - {model_name}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(np.array(sample_image[0]))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Input × Gradient
        input_grad_magnitude = jnp.sqrt(jnp.sum(input_grad_attr[0]**2, axis=-1))
        im1 = axes[0, 1].imshow(input_grad_magnitude, cmap='RdBu_r')
        axes[0, 1].set_title('Input × Gradient')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # SmoothGrad
        smooth_grad_magnitude = jnp.sqrt(jnp.sum(smooth_grad_attr[0]**2, axis=-1))
        im2 = axes[0, 2].imshow(smooth_grad_magnitude, cmap='RdBu_r')
        axes[0, 2].set_title('SmoothGrad')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # Gradient SHAP
        shap_magnitude = jnp.sqrt(jnp.sum(shap_attr[0]**2, axis=-1))
        im3 = axes[1, 0].imshow(shap_magnitude, cmap='RdBu_r')
        axes[1, 0].set_title('Gradient SHAP')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # Attribution correlation analysis
        attrs = [input_grad_magnitude.flatten(), 
                smooth_grad_magnitude.flatten(), 
                shap_magnitude.flatten()]
        attr_names = ['Input×Grad', 'SmoothGrad', 'Grad SHAP']
        
        correlation_matrix = np.corrcoef(attrs)
        im4 = axes[1, 1].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 1].set_title('Attribution Method Correlation')
        axes[1, 1].set_xticks(range(len(attr_names)))
        axes[1, 1].set_yticks(range(len(attr_names)))
        axes[1, 1].set_xticklabels(attr_names, rotation=45)
        axes[1, 1].set_yticklabels(attr_names)
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        
        # Add correlation values as text
        for j in range(len(attr_names)):
            for k in range(len(attr_names)):
                axes[1, 1].text(k, j, f'{correlation_matrix[j, k]:.2f}', 
                               ha='center', va='center', color='white' if abs(correlation_matrix[j, k]) > 0.5 else 'black')
        
        # Attribution statistics
        stats_data = []
        for attr, name in zip(attrs, attr_names):
            stats_data.append([np.mean(attr), np.std(attr), np.max(attr), np.sum(attr > np.percentile(attr, 90))])
        
        stats_data = np.array(stats_data)
        stat_names = ['Mean', 'Std', 'Max', 'Top 10% Count']
        
        x_pos = np.arange(len(attr_names))
        width = 0.2
        
        for j, stat_name in enumerate(stat_names):
            normalized_stats = stats_data[:, j] / np.max(stats_data[:, j])
            axes[1, 2].bar(x_pos + j * width, normalized_stats, width, 
                          label=stat_name, alpha=0.8)
        
        axes[1, 2].set_xlabel('Attribution Methods')
        axes[1, 2].set_ylabel('Normalized Value')
        axes[1, 2].set_title('Attribution Statistics')
        axes[1, 2].set_xticks(x_pos + width * 1.5)
        axes[1, 2].set_xticklabels(attr_names)
        axes[1, 2].legend()
        plt.tight_layout()
        
        # Save the plot
        safe_model_name = model_name.replace(" ", "_").replace("/", "_")
        filename = f"feature_attribution_{safe_model_name}.png"
        filepath = os.path.join(attribution_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved feature attribution plot: {filepath}")
        
        # Print statistics
        print(f"\n{model_name} Attribution Analysis:")
        for j, (attr, name) in enumerate(zip(attrs, attr_names)):
            print(f"  {name}: Mean={np.mean(attr):.4f}, Std={np.std(attr):.4f}")


# ===== LAYER-WISE INFORMATION FLOW ANALYSIS =====

def information_flow_analysis_all(models_list: list[nnx.Module], model_names_list: list[str],
                                test_ds_iter, layer_sequences: dict[str, list[str]], save_dir: str = None):
    """
    Analyze how information flows through different layers
    """
    print(f"\n🌊 INFORMATION FLOW ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    if save_dir is None:
        save_dir = create_viz_directory()
    
    flow_dir = os.path.join(save_dir, "information_flow")
    os.makedirs(flow_dir, exist_ok=True)
    
    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except:
        print("Error: Could not get sample batch")
        return
    
    sample_image = jnp.asarray(sample_batch['image'][0:1])
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        layers = layer_sequences.get(model_name, [])
        
        if not layers:
            print(f"No layer sequence specified for {model_name}")
            continue
        
        # Extract activations from multiple layers
        layer_activations = {}
        layer_stats = {}
        
        for layer_name in layers:
            try:
                activations = model(sample_image, training=False, 
                                  return_activations_for_layer=layer_name)
                
                if activations is not None:
                    act_array = np.array(activations)
                    layer_activations[layer_name] = act_array
                    
                    # Calculate information-theoretic measures
                    flat_activations = act_array.flatten()
                    
                    # Entropy approximation (binning)
                    hist, _ = np.histogram(flat_activations, bins=50, density=True)
                    hist = hist[hist > 0]  # Remove zero bins
                    entropy = -np.sum(hist * np.log2(hist + 1e-8))
                    
                    layer_stats[layer_name] = {
                        'mean_activation': np.mean(flat_activations),
                        'std_activation': np.std(flat_activations),
                        'sparsity': np.mean(flat_activations == 0),
                        'entropy': entropy,
                        'dynamic_range': np.max(flat_activations) - np.min(flat_activations),
                        'effective_rank': np.sum(np.linalg.svd(act_array.reshape(act_array.shape[0], -1), compute_uv=False) > 1e-6)
                    }
                    
            except Exception as e:
                print(f"Error getting activations for {layer_name}: {e}")
        
        if not layer_stats:
            print(f"No valid activations found for {model_name}")
            continue
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Information Flow Analysis - {model_name}', fontsize=16)
        
        layer_names_plot = list(layer_stats.keys())
        x_pos = range(len(layer_names_plot))
        
        # 1. Mean activation progression
        mean_acts = [layer_stats[layer]['mean_activation'] for layer in layer_names_plot]
        axes[0, 0].plot(x_pos, mean_acts, 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].set_title('Mean Activation Flow')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(layer_names_plot, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sparsity progression
        sparsities = [layer_stats[layer]['sparsity'] for layer in layer_names_plot]
        axes[0, 1].plot(x_pos, sparsities, 'o-', linewidth=2, markersize=6, color='orange')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Sparsity')
        axes[0, 1].set_title('Sparsity Evolution')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(layer_names_plot, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Entropy progression
        entropies = [layer_stats[layer]['entropy'] for layer in layer_names_plot]
        axes[0, 2].plot(x_pos, entropies, 'o-', linewidth=2, markersize=6, color='green')
        axes[0, 2].set_xlabel('Layer')
        axes[0, 2].set_ylabel('Entropy (bits)')
        axes[0, 2].set_title('Information Content')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(layer_names_plot, rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Dynamic range progression
        ranges = [layer_stats[layer]['dynamic_range'] for layer in layer_names_plot]
        axes[1, 0].plot(x_pos, ranges, 'o-', linewidth=2, markersize=6, color='red')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Dynamic Range')
        axes[1, 0].set_title('Activation Range Evolution')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(layer_names_plot, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Effective rank progression
        ranks = [layer_stats[layer]['effective_rank'] for layer in layer_names_plot]
        axes[1, 1].plot(x_pos, ranks, 'o-', linewidth=2, markersize=6, color='purple')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Effective Rank')
        axes[1, 1].set_title('Representation Complexity')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(layer_names_plot, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Multi-metric summary
        metrics = ['sparsity', 'entropy', 'dynamic_range']
        metric_colors = ['orange', 'green', 'red']
        
        for j, (metric, color) in enumerate(zip(metrics, metric_colors)):
            values = [layer_stats[layer][metric] for layer in layer_names_plot]
            # Normalize for comparison
            if max(values) > min(values):
                values_norm = [(v - min(values)) / (max(values) - min(values)) for v in values]
            else:
                values_norm = [0.5] * len(values)
            
            axes[1, 2].plot(x_pos, values_norm, 'o-', linewidth=2, markersize=4, 
                           color=color, alpha=0.7, label=metric)
        
        axes[1, 2].set_xlabel('Layer')
        axes[1, 2].set_ylabel('Normalized Value')
        axes[1, 2].set_title('Normalized Metrics Comparison')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(layer_names_plot, rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        safe_model_name = model_name.replace(" ", "_").replace("/", "_")
        filename = f"information_flow_{safe_model_name}.png"
        filepath = os.path.join(flow_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved information flow plot: {filepath}")
        
        # Print detailed statistics
        print(f"\n{model_name} Information Flow Summary:")
        print("-" * 40)
        for layer_name in layer_names_plot:
            stats = layer_stats[layer_name]
            print(f"{layer_name}:")
            print(f"  Mean Activation: {stats['mean_activation']:.4f}")
            print(f"  Sparsity: {stats['sparsity']:.3f}")
            print(f"  Entropy: {stats['entropy']:.2f} bits")
            print(f"  Dynamic Range: {stats['dynamic_range']:.4f}")
            print(f"  Effective Rank: {stats['effective_rank']}")


# ===== DECISION BOUNDARY VISUALIZATION =====

def decision_boundary_visualization_2d(models_list: list[nnx.Module], model_names_list: list[str],
                                     test_ds_iter, class_names: list[str], 
                                     num_samples: int = 200, save_dir: str = None):
    """
    Create 2D visualization of decision boundaries using PCA projection
    """
    print(f"\n🗺️ DECISION BOUNDARY VISUALIZATION ({len(models_list)} MODELS)")
    print("=" * 70)
    
    if save_dir is None:
        save_dir = create_viz_directory()
    
    boundary_dir = os.path.join(save_dir, "decision_boundaries")
    os.makedirs(boundary_dir, exist_ok=True)
    
    # Collect data and predictions
    images_list = []
    labels_list = []
    sample_count = 0
    
    for batch in test_ds_iter.as_numpy_iterator():
        if sample_count >= num_samples:
            break
        
        images = batch['image']
        labels = batch['label']
        
        images_list.append(images)
        labels_list.append(labels)
        sample_count += len(labels)
    
    # Combine all data
    all_images = np.vstack(images_list)[:num_samples]
    all_labels = np.concatenate(labels_list)[:num_samples]
    
    # Flatten images for PCA
    images_flat = all_images.reshape(len(all_images), -1)
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    images_2d = pca.fit_transform(images_flat)
    
    # Get predictions from all models
    model_predictions = {}
    model_confidences = {}
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        
        # Process in batches to avoid memory issues
        batch_size = 32
        predictions_list = []
        confidences_list = []
        
        for start_idx in range(0, len(all_images), batch_size):
            end_idx = min(start_idx + batch_size, len(all_images))
            batch_images = jnp.asarray(all_images[start_idx:end_idx])
            
            logits = model(batch_images, training=False)
            predictions = jnp.argmax(logits, axis=1)
            confidences = jnp.max(jax.nn.softmax(logits), axis=1)
            
            predictions_list.append(np.array(predictions))
            confidences_list.append(np.array(confidences))
        
        model_predictions[model_name] = np.concatenate(predictions_list)
        model_confidences[model_name] = np.concatenate(confidences_list)
    
    # Create visualization
    n_models = len(model_names_list)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Decision Boundaries in 2D PCA Space', fontsize=16)
    
    # Create a dense grid for decision boundary visualization
    x_min, x_max = images_2d[:, 0].min() - 1, images_2d[:, 0].max() + 1
    y_min, y_max = images_2d[:, 1].min() - 1, images_2d[:, 1].max() + 1
    
    # Lower resolution grid for computational efficiency
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                        np.linspace(y_min, y_max, 50))
    
    for idx, model_name in enumerate(model_names_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Scatter plot of actual data points
        unique_labels = np.unique(all_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label_idx, color in zip(unique_labels, colors):
            mask = all_labels == label_idx
            correct_mask = model_predictions[model_name] == all_labels
            
            # Correctly classified points
            correct_points = mask & correct_mask
            if np.any(correct_points):
                ax.scatter(images_2d[correct_points, 0], images_2d[correct_points, 1], 
                          c=[color], alpha=0.7, s=20, marker='o',
                          label=f'{class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"} (Correct)')
            
            # Incorrectly classified points
            incorrect_points = mask & ~correct_mask
            if np.any(incorrect_points):
                ax.scatter(images_2d[incorrect_points, 0], images_2d[incorrect_points, 1], 
                          c=[color], alpha=0.7, s=20, marker='x',
                          label=f'{class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"} (Wrong)')
        
        # Calculate accuracy for title
        accuracy = np.mean(model_predictions[model_name] == all_labels)
        ax.set_title(f'{model_name} (Acc: {accuracy:.3f})')
        ax.set_xlabel(f'PC1 (Explained Var: {pca.explained_variance_ratio_[0]:.3f})')
        ax.set_ylabel(f'PC2 (Explained Var: {pca.explained_variance_ratio_[1]:.3f})')
        
        # Add legend for first subplot only
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    plt.tight_layout()
    
    # Save the plot
    filename = "decision_boundaries_all_models.png"
    filepath = os.path.join(boundary_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved decision boundary plot: {filepath}")
    
    # Print summary statistics
    print("\nDecision Boundary Analysis Summary:")
    print("-" * 50)
    print(f"PCA Explained Variance Ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    for model_name in model_names_list:
        accuracy = np.mean(model_predictions[model_name] == all_labels)
        avg_confidence = np.mean(model_confidences[model_name])
        print(f"{model_name}: Accuracy={accuracy:.3f}, Avg Confidence={avg_confidence:.3f}")


# ===== PREDICTION RELIABILITY ANALYSIS =====

def prediction_reliability_analysis_all(models_list: list[nnx.Module], model_names_list: list[str],
                                       test_ds_iter, class_names: list[str],
                                       confidence_thresholds: list[float] = [0.5, 0.7, 0.9, 0.95],
                                       save_dir: str = None):
    """
    Analyze prediction reliability at different confidence levels
    """
    print(f"\n📊 PREDICTION RELIABILITY ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    if save_dir is None:
        save_dir = create_viz_directory()
    
    reliability_dir = os.path.join(save_dir, "reliability_analysis")
    os.makedirs(reliability_dir, exist_ok=True)
    
    # Collect all predictions and confidences
    model_data = {name: {'predictions': [], 'confidences': [], 'correct': []} 
                 for name in model_names_list}
    true_labels = []
    
    sample_count = 0
    for batch in test_ds_iter.as_numpy_iterator():
        if sample_count >= 1000:  # Limit for efficiency
            break
        
        images = jnp.asarray(batch['image'])
        labels = batch['label']
        
        for i, model_name in enumerate(model_names_list):
            model = models_list[i]
            logits = model(images, training=False)
            probs = jax.nn.softmax(logits)
            
            predictions = jnp.argmax(logits, axis=1)
            confidences = jnp.max(probs, axis=1)
            correct = predictions == labels
            
            model_data[model_name]['predictions'].extend(np.array(predictions))
            model_data[model_name]['confidences'].extend(np.array(confidences))
            model_data[model_name]['correct'].extend(np.array(correct))
        
        true_labels.extend(labels)
        sample_count += len(labels)
    
    # Convert to arrays
    true_labels = np.array(true_labels)
    for model_name in model_names_list:
        for key in model_data[model_name]:
            model_data[model_name][key] = np.array(model_data[model_name][key])
    
    # Reliability analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prediction Reliability Analysis', fontsize=16)
    
    # 1. Reliability diagrams (calibration curves)
    for model_name in model_names_list:
        confidences = model_data[model_name]['confidences']
        correct = model_data[model_name]['correct']
        
        # Bin predictions by confidence
        bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(bins)-1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(correct[mask])
                bin_conf = np.mean(confidences[mask])
                bin_count = np.sum(mask)
                
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(bin_count)
        
        axes[0, 0].plot(bin_confidences, bin_accuracies, 'o-', label=model_name, linewidth=2)
    
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    axes[0, 0].set_xlabel('Mean Predicted Probability')
    axes[0, 0].set_ylabel('Fraction of Positives')
    axes[0, 0].set_title('Reliability Diagram (Calibration)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy at different confidence thresholds
    for model_name in model_names_list:
        confidences = model_data[model_name]['confidences']
        correct = model_data[model_name]['correct']
        
        accuracies_at_thresholds = []
        coverage_at_thresholds = []
        
        for threshold in confidence_thresholds:
            high_conf_mask = confidences >= threshold
            if np.sum(high_conf_mask) > 0:
                acc_at_threshold = np.mean(correct[high_conf_mask])
                coverage = np.mean(high_conf_mask)
            else:
                acc_at_threshold = 0
                coverage = 0
            
            accuracies_at_thresholds.append(acc_at_threshold)
            coverage_at_thresholds.append(coverage)
        
        axes[0, 1].plot(confidence_thresholds, accuracies_at_thresholds, 'o-', 
                       label=f'{model_name} (Accuracy)', linewidth=2)
        axes[0, 1].plot(confidence_thresholds, coverage_at_thresholds, 's--', 
                       label=f'{model_name} (Coverage)', linewidth=2, alpha=0.7)
    
    axes[0, 1].set_xlabel('Confidence Threshold')
    axes[0, 1].set_ylabel('Accuracy / Coverage')
    axes[0, 1].set_title('Accuracy vs Coverage at Confidence Thresholds')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ECE (Expected Calibration Error) comparison
    ece_values = []
    
    for model_name in model_names_list:
        confidences = model_data[model_name]['confidences']
        correct = model_data[model_name]['correct']
        
        # Calculate ECE
        bins = np.linspace(0, 1, 11)
        ece = 0
        total_samples = len(confidences)
        
        for i in range(len(bins)-1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(correct[mask])
                bin_conf = np.mean(confidences[mask])
                bin_weight = np.sum(mask) / total_samples
                
                ece += bin_weight * abs(bin_acc - bin_conf)
        
        ece_values.append(ece)
    
    bars = axes[1, 0].bar(model_names_list, ece_values, alpha=0.8)
    axes[1, 0].set_ylabel('Expected Calibration Error')
    axes[1, 0].set_title('Model Calibration Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, ece_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Confidence distribution by correctness
    for model_name in model_names_list:
        confidences = model_data[model_name]['confidences']
        correct = model_data[model_name]['correct']
        
        correct_confidences = confidences[correct]
        incorrect_confidences = confidences[~correct]
        
        axes[1, 1].hist(correct_confidences, bins=30, alpha=0.5, density=True,
                       label=f'{model_name} (Correct)', histtype='step', linewidth=2)
        axes[1, 1].hist(incorrect_confidences, bins=30, alpha=0.5, density=True,
                       label=f'{model_name} (Incorrect)', histtype='step', linewidth=2, linestyle='--')
    
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Confidence Distribution by Correctness')
    axes[1, 1].legend()
    plt.tight_layout()
    
    # Save the plot
    filename = "prediction_reliability_analysis.png"
    filepath = os.path.join(reliability_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Prediction reliability analysis saved to: {filepath}")

    # Print detailed reliability metrics
    print("\nReliability Analysis Summary:")
    print("-" * 50)
    for i, model_name in enumerate(model_names_list):
        overall_accuracy = np.mean(model_data[model_name]['correct'])
        avg_confidence = np.mean(model_data[model_name]['confidences'])
        ece = ece_values[i]
        
        print(f"\n{model_name}:")
        print(f"  Overall Accuracy: {overall_accuracy:.3f}")
        print(f"  Average Confidence: {avg_confidence:.3f}")
        print(f"  Expected Calibration Error: {ece:.4f}")
        
        # High confidence accuracy
        high_conf_mask = model_data[model_name]['confidences'] >= 0.9
        if np.sum(high_conf_mask) > 0:
            high_conf_acc = np.mean(model_data[model_name]['correct'][high_conf_mask])
            high_conf_coverage = np.mean(high_conf_mask)
            print(f"  High Confidence (≥0.9): Accuracy={high_conf_acc:.3f}, Coverage={high_conf_coverage:.3f}")
    
    return model_data
