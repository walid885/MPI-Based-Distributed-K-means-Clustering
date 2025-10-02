# MPI-Based-Distributed-K-means-Clustering
# MPI-Based Distributed K-means Clustering on Astronomical Data

This project showcases **parallel and sequential K-means clustering** and compares their performance using real-world astronomical data. It handles end-to-end automation, from data cleaning and statistical analysis, to clustering and performance reporting—with clear visualizations.

## Features

- Automated data preprocessing, imputation, and scaling
- Statistical exploration and insight generation (class distributions, correlations, PCA)
- Sequential K-means implementation (scikit-learn)
- Distributed K-means implementation using **MPI** (mpi4py)
- Performance and cluster analysis with graphical PDF outputs
- Robust, color-coded Bash automation (`run.sh`)
- Modular, organized code structure

## Directory Structure

AstroDataset/ # Contains the input dataset (CSV)
src/
├── Comparison.py # Runs and compares all modes
├── dataCleaning.py # Cleans and explores the data
├── kmeans_sequantial.py # Sequential K-means clustering
├── kmeans_parallel.py # MPI-distributed K-means
├── run.sh # Orchestrates full pipeline
├── insights_plots/ # Visual insights (PDF)
└── kmeans_results/ # Results, labels, comparisons
README.md
req.txt # Python requirements


## Getting Started

1. **Install dependencies** (ideally in a virtual environment):
    ```
    pip install -r req.txt
    ```
    To run the parallel (MPI) version, install `mpi4py` and an MPI implementation (e.g., OpenMPI):

    ```
    pip install mpi4py
    sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
    ```

2. **Running the full pipeline:**
    Inside `src/`, simply run:
    ```
    bash run.sh
    ```

    This will:
    - Clean and preprocess the data
    - Run sequential K-means clustering
    - Run parallel K-means clustering (multiple processes)
    - Run comparison and generate performance reports

3. **See outputs:**
    - Visualizations in `insights_plots/`
    - Cluster results and performance stats in `kmeans_results/`
    - Comparison reports summarizing all results

## Key Concepts

- **K-means Clustering:** Unsupervised algorithm for grouping data points into `K` clusters based on feature similarity.
- **MPI Parallelization:** Uses distributed memory parallelism for large dataset clustering.
- **Data Cleaning:** Replaces invalid values, imputes missing data, normalizes features for algorithm requirements.
- **PCA and Elbow Method:** Used for visualization and optimal K selection.
- **Performance Benchmarking:** Tracks and visualizes the speedup from distribution.

## Customization

- Change cluster count(s) in `kmeans_sequantial.py`, `kmeans_parallel.py`, or `Comparison.py`.
- Adjust dataset path or feature columns as required.
- See detailed log and error messages in the terminal thanks to `run.sh`.

## Requirements

All dependencies are listed in `req.txt`. Core packages:
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn
- mpi4py (for distributed mode)
- OpenMPI (system-installed for parallel runs)




