from mpi4py import MPI
print(f"MPI initialized successfully. Rank: {MPI.COMM_WORLD.Get_rank()}, Size: {MPI.COMM_WORLD.Get_size()}")

