# -*- coding: utf-8 -*-
"""
Spyder Editor

Input: DHM measurement in form of folder containing unwrapped phase.bin files
Loops through all files
takes FFT on the middle profile
extracts weighted capillary wavelengths
aggregates data across all files
plots all points on scatter as very small dots

"""
import sys
sys.path.append(r'Z:\unwrapping_algorithm\USCD_python_example_code')

import binkoala
import matplotlib.pyplot as plt
import numpy as np
import os
import ffmpeg
import time

# let's plot the surface in 2D using grayscale
#plt.pcolormesh(phase)
#plt.show()
start = time.perf_counter()
# loop through all files in given folder
folder = r"Z:\wave_pool\10122021A_bc\PhaseTest\Float\Bin"
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for filename in os.scandir(folder):
    # read the phase.bin file using the module provided by Lyncee
    phase, header = binkoala.read_mat_bin(filename.path)
    # take the middle profile
    y = phase[399]
    x = np.linspace(0, 1255, num=800)
    # plot it
    ax.cla()
    ax.plot(x, y)
    # save the plot as a .png file in a specific folder
    plt.savefig(filename.path[:26]+"profVidTest\\"+filename.path[46:52]+"profile.png")

finish = time.perf_counter()
print(finish-start)

#go to the folder with all the png files and create a video
# (
#     ffmpeg
#     .input('Z:\\wave_pool\\10122021C\\profVid\\%d_profile.png', r=10)
#     .output('Z:\\wave_pool\\10122021C\\profVid\\profVid.mp4', r=120, vcodec="libx264", pix_fmt="yuv420p")
#     .run()
# )
 