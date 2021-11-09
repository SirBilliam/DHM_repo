# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 21:04:44 2021

animation of a profile and its FFT as you move from 0th to 800th row
of data in a single frame of DHM measurement

@author: bill
"""
from binkoala2 import read_bin
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import time

st = time.time()

# animation frame settings
fps = 20
init_row = 0
Nrows = 800

# animation file name
anim_file_name = "profilesOver1Frame_animation"

# choose frame
folder = "Z:\\wave_pool\\10272021D_bc\\"
frame = "temporal_nooffset\\Phase\\Float\\Bin\\01700_phase.bin"

# function that updates a plot of the profile for a given row
def ProfsFFT(row_number):
    xa, ya, z, header = read_bin(folder + frame)
    xa = xa*10**6
    ya = ya*10**6
    z = z*10**6
    y = z[row_number][0:800]
    y = gaussian_filter1d(y,3)
    yf = np.abs(rfft(y))
    N = len(xa)
    D = np.amax(xa)
    xf = rfftfreq(N,D/N)
    plot1[0].remove()
    plot2[0].remove()
    txt[0].remove()
    plot1[0], = ax1.plot(xa, y, color='blue')
    plot2[0], = ax2.plot(xf, np.log(np.abs(yf)), color='green')
    txt[0] = ax1.text(1000,70,"row: " + str(row_number))
  
xa, ya, z, header = read_bin(folder + frame)
xa = xa*10**6
ya = ya*10**6
z = z*10**6
y = z[0][0:800]
y = gaussian_filter1d(y,3)
yf = np.abs(rfft(y))
N = len(xa)
D = np.amax(xa)
xf = rfftfreq(N,D/N)
  
fig = plt.figure(figsize=(6,8), dpi=200, facecolor='gray', linewidth=2)
ax1 = fig.add_subplot(2,1,1, ylim=(-100,100))
ax1.set(xlabel='horizontal dimension (microns)', ylabel='height (microns)',
       title='profile from DHM')

ax2 = fig.add_subplot(2,1,2, xlim=(0.00,0.16), ylim=(0,10))
ax2.set(xlabel='wavenumber (?)', ylabel='amplitude (??)',
       title='FFT on DHM profile')

plot1, = [ax1.plot(xa, y)]
plot2, = [ax2.plot(xf, np.log(np.abs(yf)))]
txt = [ax1.text(1000,70,"row: 0")]

# run animation and save it as mp4
anim = animation.FuncAnimation(fig, ProfsFFT, Nrows, interval=50)
anim.save(folder + anim_file_name + '.mp4',writer='ffmpeg',fps=fps)

print(time.time() - st)