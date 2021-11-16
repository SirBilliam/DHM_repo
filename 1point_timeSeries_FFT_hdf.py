# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:50:30 2021

extract height of the central pixel from hdf5 file for each frame that have been processed:
    offset removed on offline koala machine then spatial and temporal unwrap
    done on server in python
plot time series data and corresponding FFT

@author: bill
"""

from scipy.fft import rfft, rfftfreq, irfft
from matplotlib import pyplot as plt
import numpy as np
from math import inf
from scipy import signal
from access_data import get_single_frame, get_sequence
import platform
from scipy.ndimage import gaussian_filter1d
import os
import time
import multiprocessing as mp
import concurrent.futures

st = time.time()

# sequence id
seq_id = '10272021G_bc'
exp_type = 'wave_pool'
data_file_postfix = '_roi-none_cal-true'
save_folder = 'media'
plot_name = "1point_timeFFT.pdf"

# sequence to extract (cutoff: 'time' or 'frame')
cutoff = 'frame'
init_time = 1.5
final_time = 2.5
init_frame = 0
final_frame = 10000
smoothing = False
sigma = 5
h = [] #initialize array of heights of central pixel
t = [] #initialize array of times corresponding to height at same index

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
times, frames = get_sequence(seq_id,exp_type,cutoff=cutoff,init_time=init_time,final_time=final_time,\
                          init_frame=init_frame,final_frame=final_frame)

#for each frame grab the DHM data (height) for the central pixel
def PointHeight(k):
    x, y, z = get_single_frame(seq_id,exp_type,frames[k],x_range=[400],y_range=[400],file_postfix=data_file_postfix)
    h.append(z[0][0])
    t.append(times[k])

for k in frames:
    PointHeight(k)

#h = np.array(h)

# apply high pass filter
sos = signal.butter(10,5,btype='highpass',analog=False,output='sos',fs=12800)
hfilt = signal.sosfilt(sos,h)

# initialize the figure
fig = plt.figure(figsize=(6,6), dpi=200, facecolor='gray', linewidth=2)

#plot the time series data
ax1 = fig.add_subplot(2, 1, 1)
ax1.set(xlabel='time (seconds)', ylabel='height (microns)',
        title='time series from DHM, corresponding FFT')
ax1.plot(t, h, color='green')
ax1.plot(t, hfilt, color='blue')

#perform FFT on time series
N = len(t)
D = np.amax(t)
yf = np.abs(rfft(h))
xf = rfftfreq(N, D/N)
yfilt = np.abs(rfft(hfilt))

#plot FFT
ax2 = fig.add_subplot(2, 1, 2, ylim=(-2,8), xlim=(0,1000))
ax2.set(xlabel='frequency (Hz)', ylabel='amplitude (??)')
ax2.plot(xf, np.log(yf/1000), color='green')
ax2.plot(xf, np.log(yfilt/1000), color='blue')

plt.savefig(seq_dir + save_folder + cd + plot_name)

# peaks = find_peaks(yf/1000, prominence=5)
# #print(peaks[0])
# print(xf[peaks[0]])
# print(peak_prominences(yf/1000, peaks[0]))

print(time.time()-st)