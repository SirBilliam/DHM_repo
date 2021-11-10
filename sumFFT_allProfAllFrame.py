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
import binkoala
from scipy.ndimage import gaussian_filter1d
import time
import multiprocessing as mp
import concurrent.futures
import os
import csv
import pandas as pd

# choose the DHM measurement folder with phase.bin files
folder = "Z:\\wave_pool\\10122021B_bc\\"
# initialize the data by reading from one bin file
phasePath = "offset_removed\\UnwrappedPhase\\Float\\Bin"
xa, ya, z, header = binkoala.read_bin_conv(folder+phasePath+"\\01000_phase.bin")
# xa, ya, and z are in microns
# set the x-axis for the FFT; wave numbers in this case
N = len(xa)
D = np.amax(xa)
xf = rfftfreq(N,D/N)

# generate a list of all file paths in the folder
# to be passed to FramesFFT within a thread pool
filepaths = []
for filename in os.scandir(folder+phasePath):
    filepaths.append(filename.path)
# initialize FFT results as a list (might consider making this an array instead...)
yf = []

# create a function that takes one profile from a frame
# does guassian smoothing, does FFT, and adds the data to the collection of FFT results
def ProfFFT(line):
    y = z[line][0:800]
    y = gaussian_filter1d(y,3)
    yf.append(np.abs(rfft(y)))

# create a cuntion that reads one bin file and then runs ProfFFT on each row of data
def FramesFFT(filepath):
    xa, ya, z, header = binkoala.read_bin_conv(filepath)
    for line in range(800):
        ProfFFT(line)

# run FramesFFT on every frame of the measurement in parallel using threading
st = time.time()
if __name__ == '__main__':
    mp.freeze_support()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(FramesFFT, filepaths)
print(time.time()-st)

# # save a csv file for each frame and a row in that file for each profile in the frame
# DFall = pd.DataFrame(yf)
# count = 0
# for filepath in filepaths:
#     DFall.iloc[count,0:800].to_csv[folder + filepath[42:47] + ".csv"]
#     count = count+1

# # add all FFT results together to get a single dataset representing this DHM measurement
st = time.time()
sumYF = sum(yf)/8000800
print(time.time()-st)

# save this cumulative dataset as its own csv with a row for each array in the list
st = time.time()
sumData = folder + "sumSpaceFFTData.csv"
with open(sumData, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(sumYF)
print(time.time()-st)

# # atlternatively save the data using pandas
# st = time.time()
# DFsum = pd.DataFrame(sumYF)
# DFsum.to_csv(sumData)
# print(time.time()-st)

# read the cumulative dataset from the csvfile saved earlier
# with open(sumData, 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     sumYF = []
#     for row in csvreader:
#         sumYF.append(float(row[1])/8000800)
# sumYF = np.array(sumYF[1:])

# find peaks of the sumulative dataset on a log scale in wavenumber space and print their wavelengths
peaks = find_peaks(np.log(sumYF), prominence=0.1)
print(1/xf[peaks[0]])

# plot the log-linear and the log-log of cumulative FFT for this DHM measurement
fig = plt.figure(figsize=(6,8), dpi=200, facecolor='gray', linewidth=2)
ax1 = fig.add_subplot(2,1,1)
ax1.set(xlabel='wavenumber (???log)', ylabel='amplitude (???log)',
        title='Cumulative spatial FFT')
ax1.plot(np.log(xf), np.log(sumYF))
ax2 = fig.add_subplot(2,1,2, xlim=(0,0.16))
ax2.set(xlabel='wavenumber (???)', ylabel='amplitude (???log)')
ax2.plot(xf, np.log(sumYF))

plt.savefig(folder+"sumFFT_profsAndFrames.pdf")











