# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:50:30 2021

@author: bill
"""

from scipy.fft import rfft, rfftfreq, irfft
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences
import binkoala
from scipy.ndimage import gaussian_filter1d
import os
import time
import multiprocessing as mp
import concurrent.futures

# define frames per second for this data
fps=12800
#define point to track
x = 399
y = 399
plot_name = "1point_timeFFT.pdf"

#for each frame grab the DHM data (height) for the central pixel

folder = "Z:\\wave_pool\\10272021B_bc\\"
bin_path = "temporal_nooffset\\Phase\\Float\\Bin"
h = [] #initialize array of heights of central pixel
f = [] #initialize array of frames corresponding to height at same index
filepaths = [] #initialize paths to files containing each frames phase.bin

def binPoint(filepath):
    z, header = binkoala.read_mat_bin(filepath)
    h.append(z[y][x]/3.14159)
    f.append(int(filepath[60:65]))

st = time.time()
for frame in os.scandir(folder+bin_path):
    filepaths.append(frame.path)
print(time.time()-st)

st2 = time.time()
if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(binPoint, filepaths)
print(time.time()-st2)

f = np.asarray(f)

fig = plt.figure(figsize=(6,6), dpi=150, facecolor='gray', linewidth=2)

#plot the time series data
ax1 = fig.add_subplot(2, 1, 1)
ax1.set(xlabel='time (seconds)', ylabel='height (microns)',
        title='time series from DHM, corresponding FFT')
time = f/fps
ax1.plot(time, h)

#perform FFT on time series
N = len(time)
D = np.amax(time)
yf = np.abs(rfft(h))
xf = rfftfreq(N, D/N)

#plot FFT
ax2 = fig.add_subplot(2, 1, 2, ylim=(-2,8), xlim=(0,1000))
ax2.set(xlabel='frequency (Hz)', ylabel='amplitude (??)')
ax2.plot(xf, np.log(yf/1000))

plt.savefig(folder + plot_name)

# peaks = find_peaks(yf/1000, prominence=5)
# #print(peaks[0])
# print(xf[peaks[0]])
# print(peak_prominences(yf/1000, peaks[0]))