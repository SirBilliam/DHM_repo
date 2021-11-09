# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:44:31 2021

plot a profile and corresponding spatial FFT, then animate it through time

@author: bill
"""

from binkoala2 import read_bin
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.animation import FuncAnimation
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d

st1 = time.time()

init_frame = 300
NF = 1000
fps = 15

folder = "Z:\\wave_pool\\10272021D_bc\\"
bin_path = "temporal_nooffset\\Phase\\Float\\Bin\\"
anim_name = "prof_animationSlow.mp4"

frame_num_str = str(init_frame).zfill(5)
bin_file = folder + bin_path + frame_num_str + "_phase.bin"
x, y, z, header = read_bin(bin_file)
# take the middle profile
Y = z[399]*10**6
X = x*10**6
Y = gaussian_filter1d(Y,3)
yf = np.abs(rfft(Y))
N = len(X)
D = np.amax(X)
xf = rfftfreq(N,D/N)

#initialize the figure window, axis, and plot element
fig = plt.figure(figsize=(6,8), dpi=200, facecolor='gray', linewidth=2)
ax1 = fig.add_subplot(2,1,1, ylim=(-100,100))
ax1.set(xlabel='horizontal dimension (microns)', ylabel='height (microns)',
       title='profile from DHM')

ax2 = fig.add_subplot(2,1,2, xlim=(0.00,0.16), ylim=(0,10))
ax2.set(xlabel='wavenumber (?)', ylabel='amplitude (??)',
       title='FFT on DHM profile')

plot1, = [ax1.plot(X, Y)]
plot2, = [ax2.plot(xf, np.log(np.abs(yf)))]
txt = [ax1.text(1000,70,"frame: " + frame_num_str)]

# create a function that generates a profile plot for each frame in the
# desired animation
def AniBinProfile(frame_number):
    # read the phase.bin file using the module provided by Lyncee
    frame_num_str = str(init_frame + frame_number).zfill(5)
    bin_file = folder + bin_path + frame_num_str + "_phase.bin"
    x, y, z, header = read_bin(bin_file)
    # take the middle profile
    Y = z[399]*10**6
    X = x*10**6
    Y = gaussian_filter1d(Y,3)
    yf = np.abs(rfft(Y))
    N = len(X)
    D = np.amax(X)
    xf = rfftfreq(N,D/N)
    plot1[0].remove()
    plot2[0].remove()
    txt[0].remove()
    plot1[0], = ax1.plot(X, Y, color='blue')
    plot2[0], = ax2.plot(xf, np.log(np.abs(yf)), color='green')
    txt[0] = ax1.text(1000,70,"row: " + frame_num_str)

anim = FuncAnimation(fig, AniBinProfile, NF, interval=50)

anim.save(folder + anim_name, fps=fps, extra_args=['-vcodec','libx264'])

print(time.time()-st1)