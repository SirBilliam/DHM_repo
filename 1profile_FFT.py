#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:13:44 2021

profile data contains 177 coordinate pairs and x coordinate ranges from 0-278
Nyquist says you need to sample at least twice the frequency of the highest frequency in the signal
Sampling frequency is 177/278 s/um, so the highest frequency we can reliably see is half that ~90/278
this is the wave number, so the smallest wavelength we can measure is 278/90 or 3.09um
if our signal is in time, then the below gives us peaks at the frequencies in the signal
but if our signal is in space, then it gives us the wavenumber
if we want to make conclusions about amplitude of the capillary waves, then we need to do more than the minimum Nyquist sampling
so let's do 4 instead of 2 >>> 44/278 or 6.3 um
NOTE: this final threshold value should be the same for all 10x configurations

@author: bill
"""
# if binkoala is an unknown module, then you need to run the following two lines
#import sys
#sys.path.append(r'Z:\unwrapping_algorithm\USCD_python_example_code')

from scipy.fft import rfft, rfftfreq, irfft
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import binkoala2
from scipy.ndimage import gaussian_filter1d

#extract profile from phase bin
# Jeremy's binkoala2 gives x-axis, y-axis, and z height in meters
xa, ya, z, header = binkoala2.read_bin("Z:\\wave_pool\\10272021D_bc\\temporal_nooffset\\Phase\\Float\\Bin\\00500_phase.bin")
# conver to microns
xa=xa*10**6
ya=ya*10**6
z=z*10**6
y = z[399][0:800]
#y = gaussian_filter1d(y,10)

#plot it
fig = plt.figure(figsize=(6,4), dpi=200, facecolor='gray', linewidth=2)
ax1 = fig.add_subplot(2,1,1, ylim=(-100,100))
ax1.set(xlabel='horizontal dimension (microns)', ylabel='height (microns)',
       title='profile from DHM')
ax1.plot(xa, y)

# perform positive value fast Fourier transform on the profile data and plot it with y-axis in log
N = len(xa)
D = np.amax(xa)
yf = rfft(y)
xf = rfftfreq(N,D/N)

#plot it
ax2 = fig.add_subplot(2,1,2, xlim=(0.00,0.16))
ax2.set(xlabel='wavenumber (?)', ylabel='amplitude (??)',
       title='FFT on DHM profile')
ax2.plot(xf, np.log(np.abs(yf)))

# find wavenumber peaks
biggest = np.amax(np.abs(yf))
peakInds = find_peaks(np.abs(yf), biggest/100) #indicies of the peaks of FFT
peakInds = peakInds[0]
#print(peakInds)

# location of peaks of FFT in wavenumber space
# inverse of those are the wavelengths
peakWN = [ ]
peakWL = [ ]
for i in peakInds:
    peakWN.append(xf[i])
    peakWL.append(1/xf[i])
print(peakWL)

# magnitude of wavenumber peaks
peakAmps = [ ]
for i in peakInds:
    peakAmps.append(np.abs(yf)[i])
#print(peakAmps)

# filter out points in wavenumber space that have amplitude less than a threshold 
# (it would be good to automatically set this threshold to correspond the the Niquist limit above, 6.3um wavelength)
count=0
for i in np.abs(yf):
    if i < biggest/100:
        yf[count]=0
    count=count+1

# perform an inverse FFT to get back the profile based only on the reliable wavenumber peaks
newY = irfft(yf)
# compare to measured profile as sanity check
fig2 = plt.figure(figsize=(6,4), dpi=200, facecolor='gray', linewidth=2)
ax1 = fig2.add_subplot(2,1,1,ylim=(-100,100))
ax1.set(xlabel='horizontal dimension (microns)', ylabel='height (microns)',
       title='profile from DHM')
ax1.plot(xa, newY)

# weight the amplitude of the peaks in wavenumber space by the wavenumber
peakAmpsW = [i/j for i,j in zip(peakAmps,peakWL)]
#plot the chosen wavelengths with their associated weights
ax2 = fig2.add_subplot(2,1,2)
ax2.set(xlabel='horizontal dimension (microns)', ylabel='height (microns)')
ax2.scatter(peakWL[1:],peakAmpsW[1:])