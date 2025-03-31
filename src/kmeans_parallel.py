from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import os
import json

def parallel_kmeans(X, k, max_iterations=100, tol=1e-4):
    """
    Parallel implementation of K-means clustering using MPI.
    
    Parameters:
    - X: Scaled feature matrix (numpy array)
    - k: Number of clusters
    - max_iterations: Maximum number of iterations
    - tol: Convergence tolerance
    
    Returns:
    - centroids: Final cluster centroids
    - labels: Cluster assignments for each data point
    - inertia: Sum of squared distances to centroids
    - iterations: Number of iterations performed
    """
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only the root process (rank 0) should initialize centroids
    if rank == 0:
        # Initialize centroids randomly from the data points
        indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[indices].copy()
    else:
        centroids = None
    
    # Broadcast initial centroids to all processes
    centroids = comm.bcast(centroids, root=0)
    
    # Calculate number of data points per process
    n_samples = X.shape[0]
    samples_per_process = n_samples // size
    
    # Determine start and end indices for this process's data chunk
    start_idx = rank * samples_per_process
    end_idx = start_idx + samples_per_process if rank < size - 1 else n_samples
    
    # Get the local data chunk
    X_local = X[start_idx:end_idx]
    
    # Initialize variables
    prev_centroids = np.zeros_like(centroids)
    labels_local = np.zeros(X_local.shape[0], dtype=int)
    iterations = 0
    converged = False
    
    # Main K-means loop
    while not converged and iterations < max_iterations:
        # Compute distances to centroids for local data
        distances = np.zeros((X_local.shape[0], k))
        for i in range(k):
            distances[:, i] = np.sum((X_local - centroids[i])**2, axis=1)
        
        # Assign points to nearest centroid
        labels_local = np.argmin(distances, axis=1)
        
        # Compute sum of points and count for each cluster
        local_sums = np.zeros((k, X.shape[1]))
        local_counts = np.zeros(k, dtype=int)
        
        for i in range(k):
            cluster_points = X_local[labels_local == i]
            if len(cluster_points) > 0:
                local_sums[i] = np.sum(cluster_points, axis=0)
                local_counts[i] = len(cluster_points)
        
        # Gather sums and counts from all processes
        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)
        
        comm.Reduce(local_sums, global_sums, op=MPI.SUM, root=0)
        comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
        
        # Only the root process updates centroids
        if rank == 0:
            # Store previous centroids
            prev_centroids = centroids.copy()
            
            # Update centroids
            for i in range(k):
                if global_counts[i] > 0:
                    centroids[i] = global_sums[i] / global_counts[i]
            
            # Check for convergence
            centroid_shift = np.sum((centroids - prev_centroids)**2)
            converged = centroid_shift < tol
        
        # Broadcast updated centroids and convergence status
        centroids = comm.bcast(centroids, root=0)
        converged = comm.bcast(converged, root=0)
        
        iterations += 1
    
    # Calculate local inertia (sum of squared distances to assigned centroids)
    local_inertia = 0.0
    for i in range(X_local.shape[0]):
        local_inertia += np.sum((X_local[i] - centroids[labels_local[i]])**2)
    
    # Sum local inertia across all processes
    global_inertia = comm.allreduce(local_inertia, op=MPI.SUM)
    
    # Gather all labels
    labels_counts = comm.gather(len(labels_local), root=0)
    
    if rank == 0:
        # Allocate array for all labels
        all_labels = np.zeros(n_samples, dtype=int)
        
        # Root process collects its own labels
        all_labels[start_idx:end_idx] = labels_local
        
        # Receive labels from other processes
        for i in range(1, size):
            proc_start = i * samples_per_process
            proc_end = proc_start + samples_per_process if i < size - 1 else n_samples
            proc_labels = np.zeros(proc_end - proc_start, dtype=int)
            comm.Recv(proc_labels, source=i, tag=10)
            all_labels[proc_start:proc_end] = proc_labels
    else:
        # Send labels to root process
        comm.Send(labels_local, dest=0, tag=10)
        all_labels = None
    
    # Broadcast all labels to all processes
    all_labels = comm.bcast(all_labels, root=0)
    
    return centroids, all_labels, global_inertia, iterations

def run_parallel_kmeans(k_values=[3, 4]):
    """
    Run parallel K-means for specified k values and analyze results.
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Create results directory (only in rank 0)
    if rank == 0:
        if not os.path.exists("kmeans_results"):
            os.makedirs("kmeans_results")
            print("Created directory: kmeans_results")
    
    # Wait for directory creation
    comm.Barrier()
    
    # Load and preprocess data (all processes load data)
    df = pd.read_csv('../AstroDataset/star_classification.csv')
    
    # Replace -9999 values with NaN
    df.replace(-9999.0, np.nan, inplace=True)
    
    # Impute missing values with median
    for col in ['u', 'g', 'r', 'i', 'z']:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Select features for clustering
    features = ['u', 'g', 'r', 'i', 'z']
    X = df[features].values
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Store results for different k values
    results = {}
    
    # Run parallel K-means for each k value
    for k in k_values:
        if rank == 0:
            print(f"\nRunning Parallel K-means with k={k}...")
        
        # Measure execution time
        start_time = time.time()
        
        # Run parallel K-means
        centroids, labels, inertia, iterations = parallel_kmeans(X_scaled, k)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        if rank == 0:
            print(f"Parallel K-means (k={k}) completed in {execution_time:.4f} seconds")
            print(f"Inertia: {inertia:.2f}")
            print(f"Iterations: {iterations}")
            
            # Store results
            results[k] = {
                'centroids': centroids.tolist(),
                'labels': labels.tolist(),
                'inertia': float(inertia),
                'iterations': iterations,
                'execution_time': execution_time
            }
            
            # Add cluster labels to the original dataframe for analysis
            df_with_clusters = df.copy()
            df_with_clusters[f'cluster_k{k}'] = labels
            
            # Analyze cluster distribution
            cluster_counts = df_with_clusters[f'cluster_k{k}'].value_counts()
            print(f"\nCluster Distribution (k={k}):")
            print(cluster_counts)
            
            # Save cluster distribution plot
            plt.figure(figsize=(10, 6))
            cluster_counts.plot(kind='bar')
            plt.title(f'Parallel Cluster Distribution (k={k})')
            plt.ylabel('Count')
            plt.xlabel('Cluster')
            plt.tight_layout()
            plt.savefig(f'kmeans_results/parallel_cluster_distribution_k{k}.pdf')
            plt.close()
            
            # Cross-tabulate clusters with actual classes
            class_cluster_cross = pd.crosstab(
                df_with_clusters['class'], 
                df_with_clusters[f'cluster_k{k}'],
                normalize='index'
            )
            
            print("\nClass distribution within clusters (normalized by class):")
            print(class_cluster_cross)
            
            # Save cross-tabulation as CSV
            class_cluster_cross.to_csv(f'kmeans_results/parallel_class_cluster_cross_k{k}.csv')
            
            # Visualize clusters in 2D using the first two features
            plt.figure(figsize=(12, 10))
            
            # Plot data points colored by cluster
            for cluster_id in range(k):
                cluster_points = X_scaled[labels == cluster_id]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           label=f'Cluster {cluster_id}', alpha=0.5)
            
            # Plot cluster centroids
            plt.scatter(centroids[:, 0], centroids[:, 1], 
                       s=200, marker='X', c='black', label='Centroids')
            
            plt.title(f'Parallel K-means Clustering (k={k}) - First Two Features')
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.legend()
            plt.savefig(f'kmeans_results/parallel_clusters_visualization_k{k}.pdf')
            plt.close()
    
    # Save all results (rank 0 only)
    if rank == 0:
        # Save comparison results
        with open('kmeans_results/parallel_performance_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Write text summary
        with open('kmeans_results/parallel_performance_comparison.txt', 'w') as f:
            f.write("Parallel K-means Performance Comparison\n")
            f.write("=======================================\n\n")
            for k in k_values:
                f.write(f"K={k}: Execution time = {results[k]['execution_time']:.4f}s, ")
                f.write(f"Inertia = {results[k]['inertia']:.2f}, ")
                f.write(f"Iterations = {results[k]['iterations']}\n")
        
        print("\n=========== Performance Comparison ===========")
        for k in k_values:
            print(f"K={k}: Execution time = {results[k]['execution_time']:.4f}s, ", end="")
            print(f"Inertia = {results[k]['inertia']:.2f}, ", end="")
            print(f"Iterations = {results[k]['iterations']}")
    
    return results

if __name__ == "__main__":
    # Run parallel K-means for k=3 and k=4
    results = run_parallel_kmeans(k_values=[3, 4])