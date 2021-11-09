# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:59:26 2021

@author: bill
"""
import sys
sys.path.append(r'Z:\unwrapping_algorithm\USCD_python_example_code')

import binkoala
import matplotlib.pyplot as plt
import numpy as np
import os
import ffmpeg
import time
import multiprocessing as mp
import concurrent.futures

def binSurf(filepath):
    # read the phase.bin file using the module provided by Lyncee
    phase, header = binkoala.read_mat_bin(filepath)
    # plot it
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.pcolormesh(phase, cmap='Greys')
    # save the plot as a .png file in a specific folder
    plt.savefig(filepath[:26]+"SurfVidTest\\"+filepath[46:52]+"surface.png")
    plt.close(fig)
    
# loop through all files in given folder
folder = r"Z:\wave_pool\10122021A_bc\PhaseTest\Float\Bin"
inputs = []
st1 = time.time()
for filename in os.scandir(folder):
    inputs.append(filename.path)
print(time.time()-st1)

# apply 'binSurf' in parallel to each item in 'inputs'
st2 = time.time()
if __name__ == '__main__':
    mp.freeze_support()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        outputs = executor.map(binSurf, inputs)
print(time.time()-st2)