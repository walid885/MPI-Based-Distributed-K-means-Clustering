from mpi4py import MPI
import numpy as np
import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True)
    data = df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z']].values
    return data.astype(np.float32)

if __name__ == "init":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        full_data = load_data("your_dataset.csv")
        # Split data into chunks for each process
        data_chunks = np.array_split(full_data, comm.Get_size())
    else:
        data_chunks = None
    
    local_data = comm.scatter(data_chunks, root=0)
    print(f"Rank {rank} received {len(local_data)} samples")