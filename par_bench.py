"""
script for benchmarking python parallelization
"""

import os

# number of logical processors (for serial, ncore=1)
ncore = 1

# number of processes (for serial, nproc=1)
nproc = 35

# terminal size in matrix inversion list
final_nmat = 800

# test type ("single" matrix or "list" of matrices)
type = "list"

# set environment variables
os.environ["OMP_NUM_THREADS"] = str(ncore)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncore)
os.environ["MKL_NUM_THREADS"] = str(ncore)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncore)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncore)

import numpy as np
import multiprocessing as mp
import time

def inv_rand_mat_size(x):
    ''' 
    inverts a random square matrix of int size x
    '''
    np.linalg.inv(np.random.rand(int(x), int(x)))

# matrix size iterable
if type == "list":
    data = list(range(final_nmat))
elif type == "single":
    data = [final_nmat]

# start the timer
init_time = time.time()

# process over pool
if __name__== '__main__':
    with mp.Pool(nproc) as p:
        results  = p.map(inv_rand_mat_size, data)

# print elapsed time
print('elapsed time: {0:.2f} s'.format(time.time()-init_time))