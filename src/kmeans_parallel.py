from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import time
import os
import json
import psutil

def parallel_kmeans(X, k, max_iterations=300, tol=1e-4):
    """
    Parallel K-means implementation using MPI with detailed metrics
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Initialize centroids on root
    if rank == 0:
        indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[indices].copy()
    else:
        centroids = None
    
    centroids = comm.bcast(centroids, root=0)
    
    # Data distribution
    n_samples = X.shape[0]
    samples_per_process = n_samples // size
    start_idx = rank * samples_per_process
    end_idx = start_idx + samples_per_process if rank < size - 1 else n_samples
    X_local = X[start_idx:end_idx]
    
    # Track metrics
    local_samples = X_local.shape[0]
    
    # K-means iterations
    prev_centroids = np.zeros_like(centroids)
    labels_local = np.zeros(X_local.shape[0], dtype=int)
    iterations = 0
    converged = False
    
    iteration_times = []
    
    while not converged and iterations < max_iterations:
        iter_start = time.time()
        
        # Compute distances and assign clusters
        distances = np.zeros((X_local.shape[0], k))
        for i in range(k):
            distances[:, i] = np.sum((X_local - centroids[i])**2, axis=1)
        labels_local = np.argmin(distances, axis=1)
        
        # Compute local sums and counts
        local_sums = np.zeros((k, X.shape[1]))
        local_counts = np.zeros(k, dtype=int)
        
        for i in range(k):
            cluster_points = X_local[labels_local == i]
            if len(cluster_points) > 0:
                local_sums[i] = np.sum(cluster_points, axis=0)
                local_counts[i] = len(cluster_points)
        
        # Global reduction
        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)
        comm.Reduce(local_sums, global_sums, op=MPI.SUM, root=0)
        comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
        
        # Update centroids on root
        if rank == 0:
            prev_centroids = centroids.copy()
            for i in range(k):
                if global_counts[i] > 0:
                    centroids[i] = global_sums[i] / global_counts[i]
            centroid_shift = np.sum((centroids - prev_centroids)**2)
            converged = centroid_shift < tol
        
        centroids = comm.bcast(centroids, root=0)
        converged = comm.bcast(converged, root=0)
        
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        iterations += 1
    
    # Calculate local inertia
    local_inertia = 0.0
    for i in range(X_local.shape[0]):
        local_inertia += np.sum((X_local[i] - centroids[labels_local[i]])**2)
    
    global_inertia = comm.allreduce(local_inertia, op=MPI.SUM)
    
    # Gather all labels
    labels_counts = comm.gather(len(labels_local), root=0)
    
    if rank == 0:
        all_labels = np.zeros(n_samples, dtype=int)
        all_labels[start_idx:end_idx] = labels_local
        
        for i in range(1, size):
            proc_start = i * samples_per_process
            proc_end = proc_start + samples_per_process if i < size - 1 else n_samples
            proc_labels = np.zeros(proc_end - proc_start, dtype=int)
            comm.Recv(proc_labels, source=i, tag=10)
            all_labels[proc_start:proc_end] = proc_labels
    else:
        comm.Send(labels_local, dest=0, tag=10)
        all_labels = None
    
    all_labels = comm.bcast(all_labels, root=0)
    
    # Communication metrics
    comm_overhead = {
        'avg_iteration_time': np.mean(iteration_times),
        'total_iterations': iterations,
        'samples_per_process': local_samples
    }
    
    return centroids, all_labels, global_inertia, iterations, comm_overhead

def run_parallel_kmeans(k_values=[3, 4]):
    """
    Run parallel K-means with comprehensive benchmarking
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create directories on rank 0
    if rank == 0:
        os.makedirs("kmeans_results", exist_ok=True)
        os.makedirs("benchmarks", exist_ok=True)
        
        benchmark_metrics = {
            'dataset_info': {},
            'preprocessing_metrics': {},
            'clustering_metrics': {},
            'performance_metrics': {},
            'parallel_metrics': {}
        }
    
    comm.Barrier()
    
    # Load dataset (all processes)
    load_start = time.time()
    df = pd.read_csv('../AstroDataset/star_classification.csv')
    load_time = time.time() - load_start
    
    if rank == 0:
        dataset_size = df.shape[0]
        dataset_features = df.shape[1]
        dataset_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        benchmark_metrics['dataset_info'] = {
            'total_samples': int(dataset_size),
            'total_features': int(dataset_features),
            'memory_usage_mb': round(dataset_memory_mb, 2),
            'load_time_seconds': round(load_time, 4),
            'samples_per_process': int(dataset_size // size)
        }
        
        print(f"{'='*70}")
        print(f"PARALLEL K-MEANS CLUSTERING ({size} MPI Processes)")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_size:,} samples distributed across {size} processes")
        print(f"Samples per process: ~{dataset_size // size:,}")
    
    # Preprocessing
    preprocess_start = time.time()
    df.replace(-9999.0, np.nan, inplace=True)
    
    features = ['u', 'g', 'r', 'i', 'z']
    for col in features:
        df[col].fillna(df[col].median(), inplace=True)
    
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    preprocess_time = time.time() - preprocess_start
    
    if rank == 0:
        benchmark_metrics['preprocessing_metrics'] = {
            'preprocessing_time_seconds': round(preprocess_time, 4)
        }
    
    results = {}
    
    # Run parallel K-means for each k
    for k in k_values:
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Running Parallel K-means with k={k}...")
            print(f"{'='*70}")
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)
        
        comm.Barrier()  # Synchronize all processes
        start_time = time.time()
        
        # Run parallel K-means
        centroids, labels, inertia, iterations, comm_overhead = parallel_kmeans(X_scaled, k)
        
        comm.Barrier()  # Ensure all processes finish
        execution_time = time.time() - start_time
        
        if rank == 0:
            mem_after = process.memory_info().rss / (1024 * 1024)
            memory_used = mem_after - mem_before
            
            # Quality metrics
            silhouette = silhouette_score(X_scaled, labels)
            davies_bouldin = davies_bouldin_score(X_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
            
            # Throughput
            samples_per_second = dataset_size / execution_time
            
            # Parallel efficiency
            ideal_speedup = size
            parallel_efficiency = (1.0 / execution_time) / (1.0 / execution_time * size) * 100  # Placeholder
            
            results[k] = {
                'centroids': centroids.tolist(),
                'labels': labels.tolist(),
                'inertia': float(inertia),
                'iterations': iterations,
                'execution_time': execution_time,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz,
                'memory_used_mb': memory_used,
                'throughput_samples_per_sec': samples_per_second,
                'avg_iteration_time': comm_overhead['avg_iteration_time']
            }
            
            print(f"✓ Completed in {execution_time:.4f} seconds ({samples_per_second:.0f} samples/sec)")
            print(f"  Iterations: {iterations}")
            print(f"  Avg iteration time: {comm_overhead['avg_iteration_time']:.4f} seconds")
            print(f"  Inertia: {inertia:.2f}")
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
            print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f}")
            print(f"  Memory Used: {memory_used:.2f} MB")
            
            # Cluster analysis
            df_with_clusters = df.copy()
            df_with_clusters[f'cluster_k{k}'] = labels
            
            cluster_counts = df_with_clusters[f'cluster_k{k}'].value_counts()
            cluster_sizes = cluster_counts.to_dict()
            
            balance_ratio = cluster_counts.min() / cluster_counts.max()
            
            print(f"\n  Cluster Distribution:")
            for cluster_id, count in sorted(cluster_sizes.items()):
                percentage = (count / dataset_size) * 100
                print(f"    Cluster {cluster_id}: {count:,} samples ({percentage:.1f}%)")
            
            # Save visualizations
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            cluster_counts.plot(kind='bar', ax=axes[0], color='coral')
            axes[0].set_title(f'Parallel Cluster Distribution (k={k}, {size} processes)',
                            fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Count', fontsize=12)
            axes[0].set_xlabel('Cluster ID', fontsize=12)
            axes[0].grid(axis='y', alpha=0.3)
            
            axes[1].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
                       autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
            axes[1].set_title(f'Cluster Proportions (k={k})', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'kmeans_results/parallel_cluster_distribution_k{k}.pdf', dpi=300)
            plt.close()
            
            # 2D visualization
            plt.figure(figsize=(14, 10))
            colors = plt.cm.plasma(np.linspace(0, 1, k))
            
            for cluster_id in range(k):
                cluster_points = X_scaled[labels == cluster_id]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                           label=f'Cluster {cluster_id} (n={len(cluster_points)})',
                           alpha=0.6, s=20, color=colors[cluster_id])
            
            plt.scatter(centroids[:, 0], centroids[:, 1],
                       s=300, marker='*', c='red', edgecolors='black', linewidths=2,
                       label='Centroids', zorder=10)
            
            plt.title(f'Parallel K-means Results (k={k}, {size} processes)\n'
                     f'Silhouette: {silhouette:.4f}, Time: {execution_time:.2f}s',
                     fontsize=14, fontweight='bold')
            plt.xlabel(f'{features[0]} (scaled)', fontsize=12)
            plt.ylabel(f'{features[1]} (scaled)', fontsize=12)
            plt.legend(loc='best', framealpha=0.9)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'kmeans_results/parallel_clusters_visualization_k{k}.pdf', dpi=300)
            plt.close()
            
            # Store metrics
            benchmark_metrics['clustering_metrics'][f'k_{k}'] = {
                'execution_time_seconds': round(execution_time, 4),
                'iterations': int(iterations),
                'avg_iteration_time_seconds': round(comm_overhead['avg_iteration_time'], 4),
                'inertia': round(inertia, 2),
                'silhouette_score': round(silhouette, 4),
                'davies_bouldin_index': round(davies_bouldin, 4),
                'calinski_harabasz_score': round(calinski_harabasz, 2),
                'memory_used_mb': round(memory_used, 2),
                'throughput_samples_per_sec': round(samples_per_second, 2),
                'cluster_balance_ratio': round(balance_ratio, 3),
                'cluster_sizes': cluster_sizes
            }
    
    # Save results on rank 0
    if rank == 0:
        # Parallel performance metrics
        total_runtime = sum([results[k]['execution_time'] for k in k_values])
        avg_throughput = np.mean([results[k]['throughput_samples_per_sec'] for k in k_values])
        
        benchmark_metrics['parallel_metrics'] = {
            'num_processes': size,
            'total_runtime_seconds': round(total_runtime, 4),
            'average_throughput_samples_per_sec': round(avg_throughput, 2),
            'parallelization_method': 'MPI (Message Passing Interface)',
            'communication_overhead_estimated': 'Measured per iteration'
        }
        
        # Save comprehensive benchmark
        with open('benchmarks/parallel_benchmark_report.json', 'w') as f:
            json.dump(benchmark_metrics, f, indent=4)
        
        # Save results
        with open('kmeans_results/parallel_performance_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Performance comparison table
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        print(f"{'K Value':<10} {'Time (s)':<12} {'Throughput':<18} {'Silhouette':<12}")
        print("-"*70)
        for k in k_values:
            print(f"{k:<10} {results[k]['execution_time']:<12.4f} "
                  f"{results[k]['throughput_samples_per_sec']:<18.0f} "
                  f"{results[k]['silhouette_score']:<12.4f}")
        
        # Detailed report
        with open('kmeans_results/parallel_performance_comparison.txt', 'w') as f:
            f.write("PARALLEL K-MEANS PERFORMANCE REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("PARALLEL CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Number of MPI Processes: {size}\n")
            f.write(f"Samples per Process: ~{dataset_size // size:,}\n")
            f.write(f"Total Dataset Size: {dataset_size:,} samples\n\n")
            
            f.write("CLUSTERING RESULTS\n")
            f.write("-"*80 + "\n")
            for k in k_values:
                f.write(f"\nK={k}:\n")
                f.write(f"  Execution Time: {results[k]['execution_time']:.4f} seconds\n")
                f.write(f"  Throughput: {results[k]['throughput_samples_per_sec']:.0f} samples/sec\n")
                f.write(f"  Iterations: {results[k]['iterations']}\n")
                f.write(f"  Avg Iteration Time: {results[k]['avg_iteration_time']:.4f} seconds\n")
                f.write(f"  Inertia: {results[k]['inertia']:.2f}\n")
                f.write(f"  Silhouette Score: {results[k]['silhouette_score']:.4f}\n")
                f.write(f"  Davies-Bouldin Index: {results[k]['davies_bouldin_score']:.4f}\n")
                f.write(f"  Calinski-Harabasz Score: {results[k]['calinski_harabasz_score']:.2f}\n")
                f.write(f"  Memory Used: {results[k]['memory_used_mb']:.2f} MB\n")
        
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY FOR RESUME")
        print(f"{'='*70}")
        print(f"✓ Processed {dataset_size:,} samples using {size} parallel processes")
        print(f"✓ Achieved {avg_throughput:.0f} samples/second average throughput")
        print(f"✓ Total runtime: {total_runtime:.2f} seconds")
        print(f"✓ Data distributed: ~{dataset_size // size:,} samples per process")
        print("✓ All results saved to kmeans_results/ and benchmarks/")
    
    return results

if __name__ == "__main__":
    results = run_parallel_kmeans(k_values=[3, 4])