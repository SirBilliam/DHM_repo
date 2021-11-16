# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 21:36:10 2021

Sum together spatial FFT on all profiles on all frames in a DHM measurement

@author: bill
"""
from scipy.fft import rfft, rfftfreq, irfft
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences
import access_data as ad
from scipy.ndimage import gaussian_filter1d
import time
import multiprocessing as mp
#import concurrent.futures
import os
import csv
import platform

# choose the DHM measurement and processing parameters
seq_id = '10272021A_bc'
exp_type = 'wave_pool'
data_file_postfix = '_cal_90_not_koala'
save_folder = 'media'
plot_name = "sumFFT_allProfAllFrame.pdf"

# sequence to extract (cutoff: 'time' or 'frame')
cutoff = 'frame'
init_time = 1.5
final_time = 2.5
init_frame = 0
final_frame = 1000
smoothing = False
sigma = 5

# get os type
if platform.system() == 'Darwin':
    root = '/Volumes/atom_library'
    cd = '/'
elif platform.system() == 'Windows':
    root = 'Z:'
    cd = '\\'

# workding directory
seq_dir = root + cd + exp_type + cd + seq_id + cd

if __name__ == '__main__':
    
    times, frames = ad.get_sequence(seq_id,exp_type,cutoff=cutoff,init_time=init_time,final_time=final_time,\
                              init_frame=init_frame,final_frame=final_frame)
    
    st = time.time()
    x,y,t,data = ad.get_multi_frame(seq_id, exp_type, frames, file_postfix=data_file_postfix, get_xy=True)
    print(time.time()-st)
    
    # set the x-axis for the FFT; wave numbers in this case
    N = len(x)
    D = np.amax(x)
    xf = rfftfreq(N,D/N)
    
    # initialize FFT results as a list (might consider making this an array instead...)
    yf = []
    
    #create a function that takes the FFT of every profile line in a frame
    def FramesFFT(frame):
        frame_data = data[frame]
        for line in frame_data:
            y = gaussian_filter1d(line,sigma)
            yf.append(np.abs(rfft(y)))
    
    # run FramesFFT on every frame of the measurement in parallel using processes
    st = time.time()
    for frame in frames:
        FramesFFT(frame)
    # with mp.Pool(10) as executor:
    #         executor.map(FramesFFT, frames)
    print(time.time()-st)

    # add all FFT results together to get a single dataset representing this DHM measurement
    st = time.time()
    sumYF = sum(yf)/800800
    print(time.time()-st)

# # save this cumulative dataset as its own csv with a row for each array in the list
# st = time.time()
# sumData = folder + "sumSpaceFFTData.csv"
# with open(sumData, 'w') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(sumYF)
# print(time.time()-st)

# # read the cumulative dataset from the csvfile saved earlier
# # with open(sumData, 'r') as csvfile:
# #     csvreader = csv.reader(csvfile)
# #     sumYF = []
# #     for row in csvreader:
# #         sumYF.append(float(row[1])/8000800)
# # sumYF = np.array(sumYF[1:])

# # find peaks of the cumulative dataset on a log scale in wavenumber space and print their wavelengths
# peaks = find_peaks(np.log(sumYF), prominence=0.1)
# print(1/xf[peaks[0]])

# plot the log-linear and the log-log of cumulative FFT for this DHM measurement
fig = plt.figure(figsize=(6,8), dpi=200, facecolor='gray', linewidth=2)
ax1 = fig.add_subplot(2,1,1)
ax1.set(xlabel='wavenumber (???log)', ylabel='amplitude (???log)',
        title='Cumulative spatial FFT')
ax1.plot(xf, np.log(sumYF))
ax2 = fig.add_subplot(2,1,2, xlim=(0,0.16))
ax2.set(xlabel='wavenumber (???)', ylabel='amplitude (???log)')
ax2.plot(xf, np.log(sumYF))

plt.savefig(seq_dir + save_folder + cd + plot_name)











