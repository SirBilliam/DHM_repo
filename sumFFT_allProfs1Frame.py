# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 12:31:05 2021

add up all the profile FFT results in a frame 

@author: bill
"""

from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt
import numpy as np
import binkoala2
from scipy.ndimage import gaussian_filter1d
# import time
# import multiprocessing as mp
# import concurrent.futures
# import os
# import csv
# import pandas as pd
# from scipy.signal import find_peaks, peak_prominences

# choose a frame from a DHM measurement folder
filepath = "Z:\\wave_pool\\10272021A_bc\\temporal_nooffset\\Phase\\Float\\Bin\\04500_phase.bin"   

# create a function that reads one bin file and then runs FFT on each row of data
def FramesFFT(filepath):
    yf = []
    xa, ya, z, header = binkoala2.read_bin(filepath)
    # convert to microns from meters
    xa = xa*10**6
    ya = ya*10**6
    z = z*10**6
    # set the x-axis for the FFT; wave numbers in this case
    N = len(xa)
    D = np.amax(xa)
    xf = rfftfreq(N,D/N)
    # loop through all lines of a frame
    # does guassian smoothing, does FFT, and adds the data to the collection of FFT results
    for line in range(800):
        y = z[line][0:800]
        y = gaussian_filter1d(y,3)
        yf.append(np.abs(rfft(y)))
    # add all FFT results together to get a single dataset representing this DHM measurement
    sumYF = sum(yf)/800
    # plot the log-linear and the log-log of cumulative FFT for this DHM measurement
    fig = plt.figure(figsize=(6,8), dpi=200, facecolor='gray', linewidth=2)
    ax1 = fig.add_subplot(2,1,1)
    ax1.set(xlabel='wavenumber (???log)', ylabel='amplitude (???log)',
            title='Cumulative spatial FFT')
    ax1.plot(np.log(xf), np.log(sumYF))
    ax2 = fig.add_subplot(2,1,2, xlim=(0,0.16))
    ax2.set(xlabel='wavenumber (???)', ylabel='amplitude (???log)')
    ax2.plot(xf, np.log(sumYF))

FramesFFT(filepath)

# # find peaks of the sumulative dataset on a log scale in wavenumber space and print their wavelengths
# peaks = find_peaks(np.log(sumYF), prominence=0.1)
# print(1/xf[peaks[0]])

# plt.savefig(folder+"sumFFT_allProfs1Frame.pdf")