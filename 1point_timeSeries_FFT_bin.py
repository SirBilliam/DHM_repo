# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:50:30 2021

extract height of the central pixel from phase bin files that have been processed:
    offset removed, spatial unwrapped, temporal unwrapped on offline koala machine
plot time series data and corresponding FFT

@author: bill
"""

from scipy.fft import rfft, rfftfreq, irfft
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from binkoala import read_bin_conv
import platform
from scipy.ndimage import gaussian_filter1d
import os
import time
import multiprocessing as mp
import concurrent.futures

# define frames per second for this data
fps=12800

# directories
seq_dir = "Z:\\wave_pool\\10122021A_bc\\"
bin_folder = "Time_unwrapped\\Phase\\Float\\Bin"
save_folder = 'media'
plot_name = "1point_timeFFT_bin.pdf"

h = [] #initialize array of heights of central pixel
f = [] #initialize array of frames corresponding to height at same index
inputs = []

# get iterable of paths to binfiles
for filename in os.scandir(seq_dir + bin_folder):
    inputs.append(filename.path)

# for each frame grab the DHM data (height) for the central pixel
def PointHeight(binpath):
    x, y, z, header = read_bin_conv(binpath)
    h.append(z[len(y)//2][len(x)//2])
    f.append(int(binpath[57:62]))

st2 = time.time()
if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(PointHeight, inputs)
# for binpath in inputs:
#     PointHeight(binpath)
print(time.time()-st2)

f = np.array(f)
t = f/fps # time in seconds

fig = plt.figure(figsize=(6,6), dpi=200, facecolor='gray', linewidth=2)

#plot the time series data
ax1 = fig.add_subplot(2, 1, 1)
ax1.set(xlabel='time (seconds)', ylabel='height (microns)',
        title='time series from DHM, corresponding FFT')
ax1.plot(t, h)

#perform FFT on time series
N = len(t)
D = np.amax(t)
yf = np.abs(rfft(h))
xf = rfftfreq(N, D/N)

#plot FFT
ax2 = fig.add_subplot(2, 1, 2, ylim=(-2,8), xlim=(0,1000))
ax2.set(xlabel='frequency (Hz)', ylabel='amplitude (??)')
ax2.plot(xf, np.log(yf/1000))

plt.savefig(seq_dir + save_folder + '\\' + plot_name)

# peaks = find_peaks(yf/1000, prominence=5)
# #print(peaks[0])
# print(xf[peaks[0]])
# print(peak_prominences(yf/1000, peaks[0]))