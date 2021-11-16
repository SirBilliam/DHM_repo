# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:50:30 2021

extract height of the every pixel from hdf5 file for each frame
plot time series data and corresponding FFT

@author: bill
"""

from scipy.fft import rfft, rfftfreq, irfft
from matplotlib import pyplot as plt
import numpy as np
from math import inf
from scipy.signal import find_peaks, peak_prominences
from access_data import get_single_frame, get_sequence
import platform
from scipy.ndimage import gaussian_filter1d
import os
import time
import multiprocessing as mp
import concurrent.futures
from scipy import signal

# sequence id
seq_id = '10122021C_bc'
exp_type = 'wave_pool'
data_file_postfix = '_roi-none_cal-true'
save_folder = 'media'
plot_name = "5point_timeFFT.pdf"

# sequence to extract (cutoff: 'time' or 'frame')
cutoff = 'frame'
init_time = 1.5
final_time = 2.5
init_frame = 0
final_frame = 10000
smoothing = False
sigma = 5
h = [[],[],[],[],[]]#initialize array of heights of central pixel
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
times, frames = get_sequence(seq_id,exp_type,cutoff=cutoff,init_time=init_time,\
                             final_time=final_time,init_frame=init_frame,final_frame=final_frame)

#for each frame grab the DHM data (height) for the central pixel
def AllPointHeights(k):
    x, y, z = get_single_frame(seq_id,exp_type,frames[k],get_xy=True,file_postfix=data_file_postfix)
    h[0].append(z[len(y)//2][len(x)//2])
    h[1].append(z[int(len(y)//1.5)][int(len(x)//1.5)])
    h[2].append(z[int(len(y)//1.5)][len(x)//4])
    h[3].append(z[len(y)//4][int(len(x)//1.5)])
    h[4].append(z[len(y)//4][len(x)//4])
    t.append(times[k])

st2 = time.time()
for k in frames:
    AllPointHeights(k)
print(time.time()-st2)

# apply high pass filter
sos = signal.butter(10,5,btype='highpass',analog=False,output='sos',fs=12800)
h[0] = signal.sosfilt(sos,h[0])
h[1] = signal.sosfilt(sos,h[1])
h[2] = signal.sosfilt(sos,h[2])
h[3] = signal.sosfilt(sos,h[3])
h[4] = signal.sosfilt(sos,h[4])

fig = plt.figure(figsize=(6,6), dpi=200, facecolor='gray', linewidth=2)

#plot the time series data
ax1 = fig.add_subplot(2, 1, 1)
ax1.set(xlabel='time (seconds)', ylabel='height (microns)',
        title='time series from DHM, corresponding FFT')
ax1.plot(t, h[0], color='red')
ax1.plot(t, h[1], color='purple')
ax1.plot(t, h[2], color='orange')
ax1.plot(t, h[3], color='green')
ax1.plot(t, h[4], color='blue')

#perform FFT on time series
yf = [[],[],[],[],[]]
N = len(t)
D = np.amax(t)
yf[0] = np.abs(rfft(h[0]))
yf[1] = np.abs(rfft(h[1]))
yf[2] = np.abs(rfft(h[2]))
yf[3] = np.abs(rfft(h[3]))
yf[4] = np.abs(rfft(h[4]))
sumYF = sum(yf)/5
xf = rfftfreq(N, D/N)

#plot FFT
ax2 = fig.add_subplot(2, 1, 2, ylim=(-2,8), xlim=(0,1000))
ax2.set(xlabel='frequency (Hz)', ylabel='amplitude (??)')
ax2.plot(xf, np.log(sumYF/1000))

plt.savefig(seq_dir + save_folder + cd + plot_name)

# peaks = find_peaks(yf/1000, prominence=5)
# #print(peaks[0])
# print(xf[peaks[0]])
# print(peak_prominences(yf/1000, peaks[0]))