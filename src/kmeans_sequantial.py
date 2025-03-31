import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time
import os

def run_sequential_kmeans(k_values=[3, 4]):
    # Create a directory to save results
    if not os.path.exists("kmeans_results"):
        os.makedirs("kmeans_results")
        print("Created directory: kmeans_results")
    
    # Load and preprocess the dataset
    print("Loading dataset...")
    df = pd.read_csv('../AstroDataset/star_classification.csv')
    
    # Replace -9999 values with NaN
    df.replace(-9999.0, np.nan, inplace=True)
    
    # Impute missing values with median
    for col in ['u', 'g', 'r', 'i', 'z']:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Select features for clustering
    features = ['u', 'g', 'r', 'i', 'z']
    X = df[features]
    
    # Normalize the features (important for K-means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Store results for different k values
    results = {}
    
    # Run K-means for each specified k value
    for k in k_values:
        print(f"\nRunning Sequential K-means with k={k}...")
        
        # Measure execution time
        start_time = time.time()
        
        # Initialize and fit K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Store results
        results[k] = {
            'model': kmeans,
            'labels': kmeans.labels_,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'execution_time': execution_time
        }
        
        print(f"K-means (k={k}) completed in {execution_time:.4f} seconds")
        print(f"Inertia: {kmeans.inertia_:.2f}")
        
        # Add cluster labels to the original dataframe for analysis
        df_with_clusters = df.copy()
        df_with_clusters[f'cluster_k{k}'] = kmeans.labels_
        
        # Analyze cluster distribution
        cluster_counts = df_with_clusters[f'cluster_k{k}'].value_counts()
        print(f"\nCluster Distribution (k={k}):")
        print(cluster_counts)
        
        # Save cluster distribution plot
        plt.figure(figsize=(10, 6))
        cluster_counts.plot(kind='bar')
        plt.title(f'Cluster Distribution (k={k})')
        plt.ylabel('Count')
        plt.xlabel('Cluster')
        plt.tight_layout()
        plt.savefig(f'kmeans_results/sequential_cluster_distribution_k{k}.pdf')
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
        class_cluster_cross.to_csv(f'kmeans_results/sequential_class_cluster_cross_k{k}.csv')
        
        # Visualize clusters in 2D using the first two features
        plt.figure(figsize=(12, 10))
        
        # Plot data points colored by cluster
        for cluster_id in range(k):
            cluster_points = X_scaled[kmeans.labels_ == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        label=f'Cluster {cluster_id}', alpha=0.5)
        
        # Plot cluster centroids
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   s=200, marker='X', c='black', label='Centroids')
        
        plt.title(f'K-means Clustering (k={k}) - First Two Features')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend()
        plt.savefig(f'kmeans_results/sequential_clusters_visualization_k{k}.pdf')
        plt.close()
    
    return results

if __name__ == "__main__":
    # Run sequential K-means for k=3 and k=4
    results = run_sequential_kmeans(k_values=[3, 4])
    
    # Compare the results
    print("\n=========== Performance Comparison ===========")
    print(f"K=3: Execution time = {results[3]['execution_time']:.4f}s, Inertia = {results[3]['inertia']:.2f}")
    print(f"K=4: Execution time = {results[4]['execution_time']:.4f}s, Inertia = {results[4]['inertia']:.2f}")
    
    # Save comparison results
    with open('kmeans_results/sequential_performance_comparison.txt', 'w') as f:
        f.write("Sequential K-means Performance Comparison\n")
        f.write("=========================================\n\n")
        f.write(f"K=3: Execution time = {results[3]['execution_time']:.4f}s, Inertia = {results[3]['inertia']:.2f}\n")
        f.write(f"K=4: Execution time = {results[4]['execution_time']:.4f}s, Inertia = {results[4]['inertia']:.2f}\n")