#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    """Main function to coordinate the K-means clustering process."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
 
        # Import modules (importing here avoids potential circular imports)
        import dataCleaning 
        import kmeans_sequantial
        import kmeans_parallel
        import Comparison
        
        # Load and clean the data
        print("\n==== Data Cleaning Phase ====")
        df = dataCleaning(args.data)
        
        # Perform exploratory data analysis if requested
        if args.analyze:
            print("\n==== Exploratory Data Analysis Phase ====")
            # Run EDA code from init.py or import those functions
            import init
            # You might need to explicitly run functions from init.py
            # For example: init.run_eda(df, args.features)
            
            # If using the code you provided in the paste, you'd do:
            features = args.features
            X = df[features]
            
            # Normalize the features (very important for K-means)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            # Just scale the features without analysis
            features = args.features
            X = df[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        # Find optimal K if requested
        if args.find_optimal_k:
            print("\n==== Finding Optimal K ====")
            # Run optimal K finder from init.py or import those functions
            # For example: init.find_optimal_k(X_scaled, args.max_k)
        
        # Run sequential K-means if requested
        if args.mode in ['sequential', 'both']:
            print("\n==== Running Sequential K-means ====")
            seq_results = {}
            for k in args.k_values:
                print(f"Running with K={k}...")
                start_time = time.time()
                # Call your sequential implementation
                results = kmeans_sequantial.run_kmeans(X_scaled, k, df)
                end_time = time.time()
                print(f"K={k}: Execution time = {end_time - start_time:.4f}s")
                seq_results[k] = results
        
        # Broadcast data to all processes for parallel execution
        if args.mode in ['parallel', 'both']:
            data_to_bcast = {
                'X_scaled': X_scaled,
                'k_values': args.k_values,
                'df': df
            }
        else:
            data_to_bcast = None
    else:
        data_to_bcast = None
    
    # Broadcast data for parallel execution
    if rank == 0 and args.mode in ['parallel', 'both']:
        print("\n==== Running Parallel K-means ====")
    
    data_to_bcast = comm.bcast(data_to_bcast, root=0)
    
    # Run parallel K-means if requested
    par_results = {}
    if args.mode in ['parallel', 'both'] if rank == 0 else False:
        for k in data_to_bcast['k_values']:
            if rank == 0:
                print(f"Running parallel K-means with K={k}...")
            
            # Call your parallel implementation
            start_time = time.time() if rank == 0 else None
            result = kmeans_parallel.run_kmeans_mpi(data_to_bcast['X_scaled'], k, comm, data_to_bcast['df'])
            end_time = time.time() if rank == 0 else None
            
            if rank == 0:
                print(f"K={k}: Execution time = {end_time - start_time:.4f}s")
                par_results[k] = result
    
    # Compare results if both methods were run
    if rank == 0 and args.mode == 'both':
        print("\n==== Comparing Results ====")
        Comparison.compare_results(seq_results, par_results)
        
        # Generate a detailed comparison report
        with open('comparison_results/kmeans_comparison_report.txt', 'w') as f:
            f.write("K-means Clustering Comparison Report\n")
            f.write("==================================\n\n")
            
            f.write("Sequential K-means Performance Comparison\n")
            f.write("=========================================\n\n")
            for k, result in seq_results.items():
                f.write(f"K={k}: Execution time = {result.get('execution_time', 'N/A')}s, " 
                        f"Inertia = {result.get('inertia', 'N/A')}\n")
            
            f.write("\nParallel K-means Performance Comparison\n")
            f.write("=======================================\n\n")
            for k, result in par_results.items():
                f.write(f"K={k}: Execution time = {result.get('execution_time', 'N/A')}s, "
                        f"Inertia = {result.get('inertia', 'N/A')}, "
                        f"Iterations = {result.get('iterations', 'N/A')}\n")
    
    # Wait for all processes to finish
    comm.Barrier()
    
    if rank == 0:
        print("\n==== Processing Complete ====")

if __name__ == "__main__":
    main()