# -*- coding: utf-8 -*-
"""
Spyder Editor

Input: DHM measurement in form of folder containing unwrapped phase.bin files
create a list of paths to the files in that forlder
extract bin data, plot middle profile, save plot > for each file...
in parallel

"""
import sys
sys.path.append(r'Z:\unwrapping_algorithm\USCD_python_example_code')

import binkoala
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import multiprocessing as mp
import concurrent.futures


def binProfile(filepath):
    # read the phase.bin file using the module provided by Lyncee
    phase, header = binkoala.read_mat_bin(filepath)
    # take the middle profile
    y = phase[399]
    x = np.linspace(0, 1255, num=800)
    # plot it
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    # save the plot as a .png file in a specific folder
    plt.savefig(filepath[:26]+"profVidTest\\"+filepath[46:52]+"profile.png")
    plt.close(fig)
    
# loop through all files in given folder
folder = r"Z:\wave_pool\10122021A_bc\PhaseTest\Float\Bin"
inputs = []
st1 = time.time()
for filename in os.scandir(folder):
    inputs.append(filename.path)
print(time.time()-st1)
st2 = time.time()
if __name__ == '__main__':
    mp.freeze_support()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        outputs = executor.map(binProfile, inputs)
print(time.time()-st2)
