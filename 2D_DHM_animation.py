# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:44:31 2021

building on 1D_DHM_animation, use FuncAnimation to create
a video of the phase colormap over time, i.e. a 2D animation
**this script works but its slow compared with animate_colormap or animate_surface

@author: bill
"""

# import sys
# sys.path.append(r'Z:\unwrapping_algorithm\USCD_python_example_code')

import binkoala
import matplotlib.pyplot as plt
import numpy as np
import os
import ffmpeg
import time
import multiprocessing as mp
import concurrent.futures
from matplotlib.animation import FuncAnimation

st1 = time.time()

# loop through all files in given folder and add the file paths to
# a list called 'inputs'
folder = r"Z:\wave_pool\10122021A_bc\PhaseTest\Float\Bin"
inputs = []

for filename in os.scandir(folder):
    inputs.append(filename.path)

#initialize the figure window, axis, and plot element
fig = plt.figure()
ax = plt.axes(xlim=(0, 800), ylim=(0, 800))
B, A = np.meshgrid(np.linspace(0,1255,800), np.linspace(0,1255,800))
C, h = binkoala.read_mat_bin(inputs[0])
surf = ax.pcolormesh(A,B,C,shading='gouraud')

def init():
    surf.set_array(np.asarray([]))
    return surf

# create a function that generates a profile plot for each frame in the
# desired animation
def AniBinSurface(i):
    # read the phase.bin file using the module provided by Lyncee
    C, h = binkoala.read_mat_bin(inputs[i])
    # plot it
    surf.set_array(C.ravel())
    return surf

anim = FuncAnimation(fig, AniBinSurface, init_func=init, frames=400, interval=50, blit=False, repeat=False)

anim.save(r'C:\Users\bill\Desktop\DHM\10122021A_bc\surfTest_animation.mp4', fps=30, extra_args=['-vcodec','libx264'])

print(time.time()-st1)
