# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:50:30 2021

@author: bill
"""

from scipy.fft import rfft, rfftfreq, irfft
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from access_data import get_single_frame, get_sequence
import platform
from scipy.ndimage import gaussian_filter1d
import os
import time
import multiprocessing as mp
import concurrent.futures

# define frames per second for this data
fps=12800

# sequence id
seq_id = '10122021A_bc'
exp_type = 'wave_pool'
data_file_postfix = 'full_test'
save_folder = 'media'
plot_name = "1point_timeFFT.pdf"

# sequence to extract (cutoff: 'time' or 'frame')
cutoff = 'frame'
init_time = 1.5
final_time = 2.5
init_frame = 0
final_frame = 500
smoothing = False
sigma = 5
h = [] #initialize array of heights of central pixel
t= [] #initialize array of times corresponding to height at same index

# get os type
if platform.system() == 'Darwin':
    root = '/Volumes/atom_library'
    cd = '/'
elif platform.system() == 'Windows':
    root = 'Z:'
    cd = '\\'

# workding directory
seq_dir = root + cd + exp_type + cd + seq_id + cd

# get time and frame vectors
times, frames = get_sequence(seq_id,exp_type,cutoff=cutoff,init_time=init_time,final_time=final_time, \
                         init_frame=init_frame,final_frame=final_frame)

#for each frame grab the DHM data (height) for the central pixel
def binPoint(k):
    x, y, z = get_single_frame(seq_id,exp_type,frames[k],get_xy=True,smoothing=smoothing,sigma_smooth=sigma,file_postfix=data_file_postfix)
    h.append(z[len(y)//2][len(x)//2])
    t.append(times[k])


st2 = time.time()

for k in frames:
    binPoint(k)

# if __name__ == '__main__':
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         executor.map(binPoint, frames)
print(time.time()-st2)

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

plt.savefig(seq_dir + save_folder + plot_name)

# peaks = find_peaks(yf/1000, prominence=5)
# #print(peaks[0])
# print(xf[peaks[0]])
# print(peak_prominences(yf/1000, peaks[0]))