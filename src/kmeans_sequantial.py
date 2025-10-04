import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import time
import os
import json
import psutil
import sys

def run_sequential_kmeans(k_values=[3, 4]):
    # Create directories for results
    os.makedirs("kmeans_results", exist_ok=True)
    os.makedirs("benchmarks", exist_ok=True)
    
    # Initialize benchmark metrics
    benchmark_metrics = {
        'dataset_info': {},
        'preprocessing_metrics': {},
        'clustering_metrics': {},
        'performance_metrics': {}
    }
    
    # Load dataset and measure loading time
    print("Loading dataset...")
    load_start = time.time()
    df = pd.read_csv('../AstroDataset/star_classification.csv')
    load_time = time.time() - load_start
    
    # Dataset metrics
    dataset_size = df.shape[0]
    dataset_features = df.shape[1]
    dataset_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    benchmark_metrics['dataset_info'] = {
        'total_samples': int(dataset_size),
        'total_features': int(dataset_features),
        'memory_usage_mb': round(dataset_memory_mb, 2),
        'load_time_seconds': round(load_time, 4),
        'file_size_mb': round(os.path.getsize('../AstroDataset/star_classification.csv') / (1024 * 1024), 2)
    }
    
    print(f"Dataset: {dataset_size:,} samples, {dataset_features} features, {dataset_memory_mb:.2f} MB")
    
    # Preprocessing metrics
    preprocess_start = time.time()
    
    # Count missing values
    missing_before = df.isnull().sum().sum()
    df.replace(-9999.0, np.nan, inplace=True)
    missing_after_replacement = df.isnull().sum().sum()
    
    # Impute missing values
    features = ['u', 'g', 'r', 'i', 'z']
    for col in features:
        df[col].fillna(df[col].median(), inplace=True)
    
    missing_after_imputation = df.isnull().sum().sum()
    preprocess_time = time.time() - preprocess_start
    
    benchmark_metrics['preprocessing_metrics'] = {
        'missing_values_detected': int(missing_after_replacement),
        'missing_values_imputed': int(missing_after_replacement - missing_after_imputation),
        'preprocessing_time_seconds': round(preprocess_time, 4),
        'imputation_rate': round((missing_after_replacement / (dataset_size * len(features))) * 100, 2)
    }
    
    # Feature scaling metrics
    X = df[features]
    scale_start = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scale_time = time.time() - scale_start
    
    benchmark_metrics['preprocessing_metrics']['scaling_time_seconds'] = round(scale_time, 4)
    
    # Store results for different k values
    results = {}
    
    # Run K-means for each specified k value
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Running Sequential K-means with k={k}...")
        print(f"{'='*60}")
        
        # Get memory usage before clustering
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Measure execution time
        start_time = time.time()
        
        # Initialize and fit K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X_scaled)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Get memory usage after clustering
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = mem_after - mem_before
        
        # Calculate clustering quality metrics
        silhouette = silhouette_score(X_scaled, kmeans.labels_)
        davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)
        calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans.labels_)
        
        # Calculate throughput
        samples_per_second = dataset_size / execution_time
        
        # Store results
        results[k] = {
            'model': kmeans,
            'labels': kmeans.labels_,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'execution_time': execution_time,
            'iterations': kmeans.n_iter_,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'memory_used_mb': memory_used,
            'throughput_samples_per_sec': samples_per_second
        }
        
        print(f"✓ Completed in {execution_time:.4f} seconds ({samples_per_second:.0f} samples/sec)")
        print(f"  Iterations: {kmeans.n_iter_}")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f}")
        print(f"  Memory Used: {memory_used:.2f} MB")
        
        # Add cluster labels to dataframe
        df_with_clusters = df.copy()
        df_with_clusters[f'cluster_k{k}'] = kmeans.labels_
        
        # Cluster distribution analysis
        cluster_counts = df_with_clusters[f'cluster_k{k}'].value_counts()
        cluster_sizes = cluster_counts.to_dict()
        
        # Calculate cluster balance metric
        min_cluster_size = cluster_counts.min()
        max_cluster_size = cluster_counts.max()
        balance_ratio = min_cluster_size / max_cluster_size
        
        print(f"\n  Cluster Distribution:")
        for cluster_id, count in sorted(cluster_sizes.items()):
            percentage = (count / dataset_size) * 100
            print(f"    Cluster {cluster_id}: {count:,} samples ({percentage:.1f}%)")
        print(f"  Balance Ratio: {balance_ratio:.3f}")
        
        # Save cluster distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        cluster_counts.plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title(f'Cluster Distribution (k={k})', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_xlabel('Cluster ID', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Pie chart
        axes[1].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
                   autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
        axes[1].set_title(f'Cluster Size Proportions (k={k})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'kmeans_results/sequential_cluster_distribution_k{k}.pdf', dpi=300)
        plt.close()
        
        # Cross-tabulation with actual classes
        class_cluster_cross = pd.crosstab(
            df_with_clusters['class'], 
            df_with_clusters[f'cluster_k{k}'],
            normalize='index'
        )
        
        print(f"\n  Class-Cluster Cross-tabulation:")
        print(class_cluster_cross)
        class_cluster_cross.to_csv(f'kmeans_results/sequential_class_cluster_cross_k{k}.csv')
        
        # Visualize clusters in 2D
        plt.figure(figsize=(14, 10))
        
        # Plot data points colored by cluster
        colors = plt.cm.viridis(np.linspace(0, 1, k))
        for cluster_id in range(k):
            cluster_points = X_scaled[kmeans.labels_ == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       label=f'Cluster {cluster_id} (n={len(cluster_points)})',
                       alpha=0.6, s=20, color=colors[cluster_id])
        
        # Plot cluster centroids
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   s=300, marker='*', c='red', edgecolors='black', linewidths=2,
                   label='Centroids', zorder=10)
        
        plt.title(f'K-means Clustering Results (k={k})\nSilhouette Score: {silhouette:.4f}',
                 fontsize=14, fontweight='bold')
        plt.xlabel(f'{features[0]} (scaled)', fontsize=12)
        plt.ylabel(f'{features[1]} (scaled)', fontsize=12)
        plt.legend(loc='best', framealpha=0.9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'kmeans_results/sequential_clusters_visualization_k{k}.pdf', dpi=300)
        plt.close()
        
        # Store metrics for this k value
        benchmark_metrics['clustering_metrics'][f'k_{k}'] = {
            'execution_time_seconds': round(execution_time, 4),
            'iterations': int(kmeans.n_iter_),
            'inertia': round(kmeans.inertia_, 2),
            'silhouette_score': round(silhouette, 4),
            'davies_bouldin_index': round(davies_bouldin, 4),
            'calinski_harabasz_score': round(calinski_harabasz, 2),
            'memory_used_mb': round(memory_used, 2),
            'throughput_samples_per_sec': round(samples_per_second, 2),
            'cluster_balance_ratio': round(balance_ratio, 3),
            'cluster_sizes': cluster_sizes
        }
    
    # Performance summary metrics
    total_runtime = sum([results[k]['execution_time'] for k in k_values])
    avg_throughput = np.mean([results[k]['throughput_samples_per_sec'] for k in k_values])
    
    benchmark_metrics['performance_metrics'] = {
        'total_runtime_seconds': round(total_runtime, 4),
        'average_throughput_samples_per_sec': round(avg_throughput, 2),
        'cpu_cores_used': 1,
        'parallelization': 'Sequential'
    }
    
    # Save comprehensive benchmark report
    with open('benchmarks/sequential_benchmark_report.json', 'w') as f:
        json.dump(benchmark_metrics, f, indent=4)
    
    # Generate comparison table
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'K Value':<10} {'Time (s)':<12} {'Throughput':<15} {'Silhouette':<12} {'Memory (MB)':<12}")
    print("-"*60)
    for k in k_values:
        print(f"{k:<10} {results[k]['execution_time']:<12.4f} "
              f"{results[k]['throughput_samples_per_sec']:<15.0f} "
              f"{results[k]['silhouette_score']:<12.4f} "
              f"{results[k]['memory_used_mb']:<12.2f}")
    
    # Save detailed performance report
    with open('kmeans_results/sequential_performance_comparison.txt', 'w') as f:
        f.write("SEQUENTIAL K-MEANS PERFORMANCE REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Samples: {dataset_size:,}\n")
        f.write(f"Features: {dataset_features}\n")
        f.write(f"Dataset Size: {dataset_memory_mb:.2f} MB\n")
        f.write(f"Load Time: {load_time:.4f} seconds\n\n")
        
        f.write("PREPROCESSING METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Missing Values Imputed: {missing_after_replacement:,}\n")
        f.write(f"Preprocessing Time: {preprocess_time:.4f} seconds\n")
        f.write(f"Scaling Time: {scale_time:.4f} seconds\n\n")
        
        f.write("CLUSTERING RESULTS\n")
        f.write("-"*80 + "\n")
        for k in k_values:
            f.write(f"\nK={k}:\n")
            f.write(f"  Execution Time: {results[k]['execution_time']:.4f} seconds\n")
            f.write(f"  Throughput: {results[k]['throughput_samples_per_sec']:.0f} samples/sec\n")
            f.write(f"  Iterations: {results[k]['iterations']}\n")
            f.write(f"  Inertia: {results[k]['inertia']:.2f}\n")
            f.write(f"  Silhouette Score: {results[k]['silhouette_score']:.4f}\n")
            f.write(f"  Davies-Bouldin Index: {results[k]['davies_bouldin_score']:.4f}\n")
            f.write(f"  Calinski-Harabasz Score: {results[k]['calinski_harabasz_score']:.2f}\n")
            f.write(f"  Memory Used: {results[k]['memory_used_mb']:.2f} MB\n")
    
    print("\n✓ All results saved to kmeans_results/ and benchmarks/")
    
    return results, benchmark_metrics

if __name__ == "__main__":
    print("="*60)
    print("SEQUENTIAL K-MEANS CLUSTERING WITH BENCHMARKING")
    print("="*60)
    
    results, metrics = run_sequential_kmeans(k_values=[3, 4])
    
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY FOR RESUME")
    print(f"{'='*60}")
    print(f"✓ Processed {metrics['dataset_info']['total_samples']:,} astronomical objects")
    print(f"✓ Dataset: {metrics['dataset_info']['memory_usage_mb']:.2f} MB across {metrics['dataset_info']['total_features']} features")
    print(f"✓ Achieved {metrics['performance_metrics']['average_throughput_samples_per_sec']:.0f} samples/second throughput")
    print(f"✓ Total runtime: {metrics['performance_metrics']['total_runtime_seconds']:.2f} seconds")
    print(f"✓ Imputed {metrics['preprocessing_metrics']['missing_values_imputed']:,} missing values")