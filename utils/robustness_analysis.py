import typing as tp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
import scipy.stats
from sklearn.metrics import precision_recall_curve, roc_curve, auc

Array = jax.Array

# ===== ADVERSARIAL ROBUSTNESS ANALYSIS =====

def adversarial_robustness_analysis_all(models_list: list[nnx.Module], model_names_list: list[str],
                                      test_ds_iter, class_names: list[str],
                                      epsilon_values: list[float] = [0.01, 0.03, 0.05, 0.1]):
    """
    Test model robustness against adversarial examples using FGSM attack
    """
    print(f"\n⚔️ ADVERSARIAL ROBUSTNESS ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except:
        print("Error: Could not get sample batch for adversarial testing")
        return
    
    original_images = jnp.asarray(sample_batch['image'][:50])  # Test on subset
    true_labels = sample_batch['label'][:50]
    
    results = {model_name: {'clean_acc': 0, 'adv_acc': {}} for model_name in model_names_list}
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        
        # Test clean accuracy
        clean_logits = model(original_images, training=False)
        clean_predictions = jnp.argmax(clean_logits, axis=1)
        clean_accuracy = jnp.mean(clean_predictions == true_labels)
        results[model_name]['clean_acc'] = float(clean_accuracy)
        
        # FGSM Attack function
        @jax.jit
        def fgsm_attack(images, labels, epsilon):
            def loss_fn(imgs):
                logits = model(imgs, training=False)
                return -jnp.mean(jax.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels))
            
            gradients = jax.grad(loss_fn)(images)
            # Normalize gradients
            gradients = gradients / (jnp.linalg.norm(gradients, axis=(1,2,3), keepdims=True) + 1e-8)
            adversarial_images = images + epsilon * gradients
            # Clip to valid range [0, 1]
            adversarial_images = jnp.clip(adversarial_images, 0, 1)
            return adversarial_images
        
        # Test different epsilon values
        for epsilon in epsilon_values:
            adversarial_images = fgsm_attack(original_images, true_labels, epsilon)
            adv_logits = model(adversarial_images, training=False)
            adv_predictions = jnp.argmax(adv_logits, axis=1)
            adv_accuracy = jnp.mean(adv_predictions == true_labels)
            results[model_name]['adv_acc'][epsilon] = float(adv_accuracy)
        
        print(f"{model_name} - Clean Accuracy: {clean_accuracy:.3f}")
        for epsilon in epsilon_values:
            adv_acc = results[model_name]['adv_acc'][epsilon]
            print(f"  ε={epsilon}: {adv_acc:.3f} (drop: {clean_accuracy-adv_acc:.3f})")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Adversarial Robustness Analysis', fontsize=16)
    
    # Plot robustness curves
    for model_name in model_names_list:
        epsilons = [0] + epsilon_values
        accuracies = [results[model_name]['clean_acc']] + \
                    [results[model_name]['adv_acc'][eps] for eps in epsilon_values]
        ax1.plot(epsilons, accuracies, 'o-', label=model_name, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Perturbation Strength (ε)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Robustness vs Perturbation Strength')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar chart comparing robustness at specific epsilon
    test_epsilon = epsilon_values[len(epsilon_values)//2]  # Middle epsilon value
    model_names_plot = list(model_names_list)
    clean_accs = [results[name]['clean_acc'] for name in model_names_plot]
    adv_accs = [results[name]['adv_acc'][test_epsilon] for name in model_names_plot]
    
    x_pos = np.arange(len(model_names_plot))
    width = 0.35
    
    ax2.bar(x_pos - width/2, clean_accs, width, label='Clean', alpha=0.8)
    ax2.bar(x_pos + width/2, adv_accs, width, label=f'Adversarial (ε={test_epsilon})', alpha=0.8)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Clean vs Adversarial Accuracy (ε={test_epsilon})')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names_plot)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results


def noise_robustness_analysis_all(models_list: list[nnx.Module], model_names_list: list[str],
                                test_ds_iter, class_names: list[str],
                                noise_levels: list[float] = [0.01, 0.05, 0.1, 0.2]):
    """
    Test model robustness against various types of noise
    """
    print(f"\n🌪️ NOISE ROBUSTNESS ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except:
        print("Error: Could not get sample batch for noise testing")
        return
    
    original_images = jnp.asarray(sample_batch['image'][:100])
    true_labels = sample_batch['label'][:100]
    
    noise_types = ['gaussian', 'uniform', 'salt_pepper']
    results = {model_name: {noise_type: {} for noise_type in noise_types} 
              for model_name in model_names_list}
    
    # Add clean accuracy
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        clean_logits = model(original_images, training=False)
        clean_predictions = jnp.argmax(clean_logits, axis=1)
        clean_accuracy = float(jnp.mean(clean_predictions == true_labels))
        for noise_type in noise_types:
            results[model_name][noise_type]['clean'] = clean_accuracy
    
    key = jax.random.PRNGKey(42)
    
    for noise_level in noise_levels:
        print(f"\nTesting noise level: {noise_level}")
        
        for noise_type in noise_types:
            print(f"  {noise_type} noise...")
            
            # Generate noisy images
            if noise_type == 'gaussian':
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, original_images.shape) * noise_level
                noisy_images = jnp.clip(original_images + noise, 0, 1)
            
            elif noise_type == 'uniform':
                key, subkey = jax.random.split(key)
                noise = (jax.random.uniform(subkey, original_images.shape) - 0.5) * 2 * noise_level
                noisy_images = jnp.clip(original_images + noise, 0, 1)
            
            elif noise_type == 'salt_pepper':
                key, subkey = jax.random.split(key)
                mask = jax.random.uniform(subkey, original_images.shape[:3]) < noise_level
                key, subkey = jax.random.split(key)
                salt_pepper = jax.random.uniform(subkey, original_images.shape[:3]) > 0.5
                
                noisy_images = original_images.copy()
                # Apply salt and pepper noise
                noisy_images = jnp.where(mask[..., None] & salt_pepper[..., None], 1.0, noisy_images)
                noisy_images = jnp.where(mask[..., None] & ~salt_pepper[..., None], 0.0, noisy_images)
            
            # Test all models on this noisy version
            for i, model_name in enumerate(model_names_list):
                model = models_list[i]
                noisy_logits = model(noisy_images, training=False)
                noisy_predictions = jnp.argmax(noisy_logits, axis=1)
                noisy_accuracy = float(jnp.mean(noisy_predictions == true_labels))
                results[model_name][noise_type][noise_level] = noisy_accuracy
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Noise Robustness Analysis', fontsize=16)
    
    # Plot for each noise type
    for idx, noise_type in enumerate(noise_types):
        ax = axes[idx//2, idx%2] if idx < 3 else axes[1, 1]
        
        for model_name in model_names_list:
            noise_levels_plot = ['clean'] + noise_levels
            accuracies = [results[model_name][noise_type]['clean']] + \
                        [results[model_name][noise_type][level] for level in noise_levels]
            
            x_values = [0] + list(noise_levels)
            ax.plot(x_values, accuracies, 'o-', label=model_name, linewidth=2, markersize=6)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{noise_type.replace("_", " ").title()} Noise Robustness')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Summary comparison at highest noise level
    if len(noise_types) == 3:  # If we have exactly 3 noise types, use the 4th subplot
        ax = axes[1, 1]
        
        test_noise_level = noise_levels[-1]  # Highest noise level
        model_names_plot = list(model_names_list)
        
        x_pos = np.arange(len(model_names_plot))
        width = 0.25
        
        for i, noise_type in enumerate(noise_types):
            accs = [results[name][noise_type][test_noise_level] for name in model_names_plot]
            ax.bar(x_pos + i * width, accs, width, label=f'{noise_type}', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Noise Robustness Comparison (level={test_noise_level})')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(model_names_plot)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nNoise Robustness Summary:")
    print("-" * 50)
    for model_name in model_names_list:
        print(f"\n{model_name}:")
        for noise_type in noise_types:
            clean_acc = results[model_name][noise_type]['clean']
            worst_acc = min([results[model_name][noise_type][level] for level in noise_levels])
            print(f"  {noise_type}: {clean_acc:.3f} → {worst_acc:.3f} (drop: {clean_acc-worst_acc:.3f})")
    
    return results


# ===== DATA DISTRIBUTION ANALYSIS =====

def out_of_distribution_detection_all(models_list: list[nnx.Module], model_names_list: list[str],
                                     in_distribution_ds, out_distribution_ds, 
                                     class_names: list[str]):
    """
    Analyze model behavior on out-of-distribution data
    """
    print(f"\n🎭 OUT-OF-DISTRIBUTION DETECTION ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    # Collect predictions and confidence scores
    model_results = {name: {'in_dist': {'confidences': [], 'entropies': []},
                           'out_dist': {'confidences': [], 'entropies': []}}
                    for name in model_names_list}
    
    # Process in-distribution data
    print("Processing in-distribution data...")
    sample_count = 0
    for batch in in_distribution_ds.as_numpy_iterator():
        if sample_count >= 500:  # Limit samples for efficiency
            break
        
        images = jnp.asarray(batch['image'])
        
        for i, model_name in enumerate(model_names_list):
            model = models_list[i]
            logits = model(images, training=False)
            probs = jax.nn.softmax(logits)
            
            confidences = jnp.max(probs, axis=1)
            entropies = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=1)
            
            model_results[model_name]['in_dist']['confidences'].extend(np.array(confidences))
            model_results[model_name]['in_dist']['entropies'].extend(np.array(entropies))
        
        sample_count += len(batch['image'])
    
    # Process out-of-distribution data
    print("Processing out-of-distribution data...")
    sample_count = 0
    for batch in out_distribution_ds.as_numpy_iterator():
        if sample_count >= 500:
            break
        
        images = jnp.asarray(batch['image'])
        
        for i, model_name in enumerate(model_names_list):
            model = models_list[i]
            logits = model(images, training=False)
            probs = jax.nn.softmax(logits)
            
            confidences = jnp.max(probs, axis=1)
            entropies = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=1)
            
            model_results[model_name]['out_dist']['confidences'].extend(np.array(confidences))
            model_results[model_name]['out_dist']['entropies'].extend(np.array(entropies))
        
        sample_count += len(batch['image'])
    
    # Convert to arrays
    for model_name in model_names_list:
        for dist_type in ['in_dist', 'out_dist']:
            for metric in ['confidences', 'entropies']:
                model_results[model_name][dist_type][metric] = \
                    np.array(model_results[model_name][dist_type][metric])
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Out-of-Distribution Detection Analysis', fontsize=16)
    
    # 1. Confidence distributions
    for model_name in model_names_list:
        in_conf = model_results[model_name]['in_dist']['confidences']
        out_conf = model_results[model_name]['out_dist']['confidences']
        
        axes[0, 0].hist(in_conf, alpha=0.5, label=f'{model_name} (In-Dist)', 
                       bins=30, density=True)
        axes[0, 0].hist(out_conf, alpha=0.5, label=f'{model_name} (OOD)', 
                       bins=30, density=True)
    
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Confidence Distributions')
    axes[0, 0].legend()
    
    # 2. Entropy distributions
    for model_name in model_names_list:
        in_ent = model_results[model_name]['in_dist']['entropies']
        out_ent = model_results[model_name]['out_dist']['entropies']
        
        axes[0, 1].hist(in_ent, alpha=0.5, label=f'{model_name} (In-Dist)', 
                       bins=30, density=True)
        axes[0, 1].hist(out_ent, alpha=0.5, label=f'{model_name} (OOD)', 
                       bins=30, density=True)
    
    axes[0, 1].set_xlabel('Entropy')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Entropy Distributions')
    axes[0, 1].legend()
    
    # 3. ROC curves for OOD detection using confidence
    for model_name in model_names_list:
        in_conf = model_results[model_name]['in_dist']['confidences']
        out_conf = model_results[model_name]['out_dist']['confidences']
        
        # Create labels (1 for in-dist, 0 for OOD)
        labels = np.concatenate([np.ones(len(in_conf)), np.zeros(len(out_conf))])
        scores = np.concatenate([in_conf, out_conf])  # Higher confidence = more likely in-dist
        
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        axes[1, 0].plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})', linewidth=2)
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curves (OOD Detection via Confidence)')
    axes[1, 0].legend()
    
    # 4. Statistical separation metrics
    separation_metrics = []
    model_names_plot = []
    
    for model_name in model_names_list:
        in_conf = model_results[model_name]['in_dist']['confidences']
        out_conf = model_results[model_name]['out_dist']['confidences']
        in_ent = model_results[model_name]['in_dist']['entropies']
        out_ent = model_results[model_name]['out_dist']['entropies']
        
        # KL divergence approximation
        conf_separation = np.abs(np.mean(in_conf) - np.mean(out_conf))
        ent_separation = np.abs(np.mean(in_ent) - np.mean(out_ent))
        
        # Statistical test
        _, conf_p_value = scipy.stats.mannwhitneyu(in_conf, out_conf)
        _, ent_p_value = scipy.stats.mannwhitneyu(in_ent, out_ent)
        
        separation_metrics.append([conf_separation, ent_separation, 
                                 -np.log10(conf_p_value + 1e-300)])
        model_names_plot.append(model_name)
    
    separation_metrics = np.array(separation_metrics)
    metric_names = ['Conf. Separation', 'Entropy Separation', '-log10(p-value)']
    
    x_pos = np.arange(len(model_names_plot))
    width = 0.25
    
    for i, metric_name in enumerate(metric_names):
        axes[1, 1].bar(x_pos + i * width, separation_metrics[:, i], 
                      width, label=metric_name, alpha=0.8)
    
    axes[1, 1].set_title('OOD Detection Performance Metrics')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Metric Value')
    axes[1, 1].set_xticks(x_pos + width)
    axes[1, 1].set_xticklabels(model_names_plot)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\nOOD Detection Summary:")
    print("-" * 50)
    for model_name in model_names_list:
        in_conf_mean = np.mean(model_results[model_name]['in_dist']['confidences'])
        out_conf_mean = np.mean(model_results[model_name]['out_dist']['confidences'])
        in_ent_mean = np.mean(model_results[model_name]['in_dist']['entropies'])
        out_ent_mean = np.mean(model_results[model_name]['out_dist']['entropies'])
        
        print(f"\n{model_name}:")
        print(f"  In-Dist: Conf={in_conf_mean:.3f}, Entropy={in_ent_mean:.3f}")
        print(f"  OOD:     Conf={out_conf_mean:.3f}, Entropy={out_ent_mean:.3f}")
        print(f"  Separation: Conf={abs(in_conf_mean-out_conf_mean):.3f}, "
              f"Entropy={abs(in_ent_mean-out_ent_mean):.3f}")
    
    return model_results


# ===== PERFORMANCE PROFILING =====

def computational_efficiency_analysis_all(models_list: list[nnx.Module], model_names_list: list[str],
                                         test_ds_iter, batch_sizes: list[int] = [1, 8, 32, 64]):
    """
    Analyze computational efficiency and memory usage of models
    """
    print(f"\n⚡ COMPUTATIONAL EFFICIENCY ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    try:
        sample_batch = next(test_ds_iter.as_numpy_iterator())
    except:
        print("Error: Could not get sample batch for efficiency testing")
        return
    
    sample_image = jnp.asarray(sample_batch['image'][0:1])
    
    efficiency_results = {name: {'inference_times': {}, 'memory_usage': {}, 'flops': {}}
                         for name in model_names_list}
    
    import time
    import tracemalloc
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create batch of appropriate size
        test_batch = jnp.repeat(sample_image, batch_size, axis=0)
        
        for i, model_name in enumerate(model_names_list):
            model = models_list[i]
            
            # Warm up
            for _ in range(5):
                _ = model(test_batch, training=False)
            
            # Time inference
            times = []
            for _ in range(10):  # Multiple runs for averaging
                start_time = time.perf_counter()
                _ = model(test_batch, training=False)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time  # samples per second
            
            efficiency_results[model_name]['inference_times'][batch_size] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'throughput': throughput
            }
            
            print(f"  {model_name} (bs={batch_size}): {avg_time*1000:.2f}±{std_time*1000:.2f}ms, "
                  f"{throughput:.1f} samples/sec")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Computational Efficiency Analysis', fontsize=16)
    
    # 1. Inference time vs batch size
    for model_name in model_names_list:
        times = [efficiency_results[model_name]['inference_times'][bs]['avg_time'] * 1000 
                for bs in batch_sizes]
        axes[0, 0].plot(batch_sizes, times, 'o-', label=model_name, linewidth=2, markersize=6)
    
    axes[0, 0].set_xlabel('Batch Size')
    axes[0, 0].set_ylabel('Inference Time (ms)')
    axes[0, 0].set_title('Inference Time vs Batch Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log', base=2)
    
    # 2. Throughput vs batch size
    for model_name in model_names_list:
        throughputs = [efficiency_results[model_name]['inference_times'][bs]['throughput'] 
                      for bs in batch_sizes]
        axes[0, 1].plot(batch_sizes, throughputs, 'o-', label=model_name, linewidth=2, markersize=6)
    
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Throughput (samples/sec)')
    axes[0, 1].set_title('Throughput vs Batch Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log', base=2)
    
    # 3. Time per sample vs batch size
    for model_name in model_names_list:
        time_per_sample = [efficiency_results[model_name]['inference_times'][bs]['avg_time'] / bs * 1000
                          for bs in batch_sizes]
        axes[1, 0].plot(batch_sizes, time_per_sample, 'o-', label=model_name, linewidth=2, markersize=6)
    
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('Time per Sample (ms)')
    axes[1, 0].set_title('Time per Sample vs Batch Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log', base=2)
    
    # 4. Efficiency comparison at optimal batch size
    optimal_bs = batch_sizes[-1]  # Usually largest batch size is most efficient
    model_names_plot = list(model_names_list)
    
    throughputs_comp = [efficiency_results[name]['inference_times'][optimal_bs]['throughput']
                       for name in model_names_plot]
    times_comp = [efficiency_results[name]['inference_times'][optimal_bs]['avg_time'] * 1000
                 for name in model_names_plot]
    
    x_pos = np.arange(len(model_names_plot))
    
    # Normalize for comparison (higher is better for throughput, lower is better for time)
    throughputs_norm = np.array(throughputs_comp) / max(throughputs_comp)
    times_norm = min(times_comp) / np.array(times_comp)
    
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, throughputs_norm, width, label='Relative Throughput', alpha=0.8)
    axes[1, 1].bar(x_pos + width/2, times_norm, width, label='Relative Speed', alpha=0.8)
    
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Normalized Performance')
    axes[1, 1].set_title(f'Performance Comparison (batch size={optimal_bs})')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(model_names_plot)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return efficiency_results


# ===== MODEL COMPLEXITY ANALYSIS =====

def model_complexity_analysis_all(models_list: list[nnx.Module], model_names_list: list[str]):
    """
    Analyze model complexity in terms of parameters, layers, and architecture
    """
    print(f"\n🏗️ MODEL COMPLEXITY ANALYSIS ({len(models_list)} MODELS)")
    print("=" * 70)
    
    complexity_results = {}
    
    for i, model_name in enumerate(model_names_list):
        model = models_list[i]
        
        # Count parameters
        total_params = 0
        trainable_params = 0
        layer_count = 0
        
        def count_params_recursive(module, prefix=""):
            nonlocal total_params, trainable_params, layer_count
            
            if hasattr(module, '__dict__'):
                for name, attr in module.__dict__.items():
                    if isinstance(attr, jnp.ndarray):
                        # This is a parameter
                        param_count = attr.size
                        total_params += param_count
                        trainable_params += param_count  # Assume all are trainable
                        
                    elif hasattr(attr, '__dict__'):
                        # This might be a sub-module
                        layer_count += 1
                        count_params_recursive(attr, f"{prefix}.{name}" if prefix else name)
        
        count_params_recursive(model)
        
        # Calculate model size in MB (assuming float32)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        complexity_results[model_name] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layer_count': layer_count,
            'model_size_mb': model_size_mb
        }
        
        print(f"{model_name}:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Estimated Layers: {layer_count}")
        print(f"  Model Size: {model_size_mb:.2f} MB")
        print()
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Complexity Analysis', fontsize=16)
    
    model_names_plot = list(model_names_list)
    
    # 1. Parameter count comparison
    param_counts = [complexity_results[name]['total_params'] for name in model_names_plot]
    axes[0, 0].bar(model_names_plot, param_counts, alpha=0.8)
    axes[0, 0].set_ylabel('Number of Parameters')
    axes[0, 0].set_title('Total Parameters')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(param_counts):
        axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom')
    
    # 2. Model size comparison
    model_sizes = [complexity_results[name]['model_size_mb'] for name in model_names_plot]
    axes[0, 1].bar(model_names_plot, model_sizes, alpha=0.8, color='orange')
    axes[0, 1].set_ylabel('Model Size (MB)')
    axes[0, 1].set_title('Model Size')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(model_sizes):
        axes[0, 1].text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    # 3. Parameters vs Performance (if we have accuracy data)
    # This would require accuracy data from previous analysis
    axes[1, 0].scatter(param_counts, [1] * len(param_counts), alpha=0.7, s=100)
    for i, name in enumerate(model_names_plot):
        axes[1, 0].annotate(name, (param_counts[i], 1), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 0].set_xlabel('Number of Parameters')
    axes[1, 0].set_ylabel('Relative Position')
    axes[1, 0].set_title('Parameter Count Distribution')
    axes[1, 0].set_ylim(0.5, 1.5)
    
    # 4. Complexity metrics radar chart (simplified)
    metrics = ['params', 'size', 'layers']
    metric_values = []
    
    for name in model_names_plot:
        # Normalize metrics to 0-1 scale
        params_norm = complexity_results[name]['total_params'] / max(param_counts)
        size_norm = complexity_results[name]['model_size_mb'] / max(model_sizes)
        layers_norm = complexity_results[name]['layer_count'] / max([complexity_results[n]['layer_count'] for n in model_names_plot])
        
        metric_values.append([params_norm, size_norm, layers_norm])
    
    metric_values = np.array(metric_values)
    
    x_pos = np.arange(len(metrics))
    width = 0.8 / len(model_names_plot)
    
    for i, model_name in enumerate(model_names_plot):
        axes[1, 1].bar(x_pos + i * width, metric_values[i], width, 
                      label=model_name, alpha=0.8)
    
    axes[1, 1].set_xlabel('Complexity Metrics')
    axes[1, 1].set_ylabel('Normalized Value')
    axes[1, 1].set_title('Normalized Complexity Comparison')
    axes[1, 1].set_xticks(x_pos + width * (len(model_names_plot) - 1) / 2)
    axes[1, 1].set_xticklabels(['Parameters', 'Size (MB)', 'Layers'])
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return complexity_results
