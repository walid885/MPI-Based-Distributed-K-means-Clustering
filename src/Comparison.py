import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import subprocess
import json

def run_comparison():
    """
    Run both sequential and parallel K-means implementations and compare results
    """
    # Create results directory
    if not os.path.exists("comparison_results"):
        os.makedirs("comparison_results")
        print("Created directory: comparison_results")
    
    # Run sequential implementation
    print("\n============ Running Sequential K-means ============")
    start_time = time.time()
    subprocess.run(["python3", "kmeans_sequantial.py"], check=True)
    seq_total_time = time.time() - start_time
    
    # Run parallel implementation with different numbers of processes
    print("\n============ Running Parallel K-means ============")
    parallel_times = {}
    
    for num_procs in [2, 4, 6]:  # Adjust based on your system's capabilities
        if num_procs <= os.cpu_count():
            print(f"\nRunning with {num_procs} processes:")
            start_time = time.time()
            subprocess.run(["mpirun", "-n", str(num_procs), "python3", "kmeans_parallel.py"], check=True)
            parallel_times[num_procs] = time.time() - start_time
    
    # Load results
    with open('kmeans_results/sequential_performance_comparison.txt', 'r') as f:
        sequential_results = f.read()
    
    with open('kmeans_results/parallel_performance_results.json', 'r') as f:
        parallel_results = json.load(f)
    
    # Extract execution times for comparison
    seq_times = {}
    for k in [3, 4]:
        # Extract from sequential results text
        seq_times[k] = None
        for line in sequential_results.split('\n'):
            if f"K={k}: Execution time" in line:
                seq_times[k] = float(line.split("Execution time = ")[1].split("s")[0])
    
    # Create comparison report
    with open('comparison_results/kmeans_comparison_report.txt', 'w') as f:
        f.write("K-means Clustering Performance Comparison\n")
        f.write("========================================\n\n")
        
        f.write("1. Total Runtime Comparison:\n")
        f.write(f"   Sequential implementation: {seq_total_time:.4f} seconds\n")
        for procs, time_taken in parallel_times.items():
            speedup = seq_total_time / time_taken
            f.write(f"   Parallel ({procs} processes): {time_taken:.4f} seconds (Speedup: {speedup:.2f}x)\n")
        
        f.write("\n2. Individual K-means Runs:\n")
        for k in [3, 4]:
            f.write(f"\n   K={k}:\n")
            f.write(f"   Sequential: {seq_times[k]:.4f} seconds, Inertia: {parallel_results[str(k)]['inertia']:.2f}\n")
            f.write(f"   Parallel: {parallel_results[str(k)]['execution_time']:.4f} seconds, ")
            f.write(f"Inertia: {parallel_results[str(k)]['inertia']:.2f}, ")
            f.write(f"Iterations: {parallel_results[str(k)]['iterations']}\n")
            
            speedup = seq_times[k] / parallel_results[str(k)]['execution_time']
            f.write(f"   Speedup: {speedup:.2f}x\n")
    
    # Create visualization of speedup
    plt.figure(figsize=(12, 8))
    
    # Plot total runtime comparison
    plt.subplot(2, 1, 1)
    runtimes = [seq_total_time] + [parallel_times[p] for p in sorted(parallel_times.keys())]
    labels = ['Sequential'] + [f'Parallel ({p} procs)' for p in sorted(parallel_times.keys())]
    plt.bar(labels, runtimes, color=['blue'] + ['green'] * len(parallel_times))
    plt.ylabel('Total Runtime (seconds)')
    plt.title('Total Runtime Comparison')
    
    # Plot speedup by k value
    plt.subplot(2, 1, 2)
    process_counts = sorted(parallel_times.keys())
    speedups_k3 = []
    speedups_k4 = []
    
    for procs in process_counts:
        # This is an approximation since we don't have per-process timing for each k value
        speedups_k3.append(seq_times[3] / parallel_results['3']['execution_time'])
        speedups_k4.append(seq_times[4] / parallel_results['4']['execution_time'])
    
    plt.plot(process_counts, speedups_k3, 'o-', label='K=3')
    plt.plot(process_counts, speedups_k4, 's-', label='K=4')
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup (Sequential/Parallel)')
    plt.title('Speedup by Cluster Count (K)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_results/performance_comparison.pdf')
    plt.close()
    
    print("\nComparison complete. Results saved to comparison_results/")

if __name__ == "__main__":
    run_comparison()