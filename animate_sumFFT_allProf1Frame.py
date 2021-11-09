# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 13:01:17 2021

create an animation that applies sumFFT_allProfs1Frame to each frame of a measurement

@author: bill
"""

from binkoala2 import read_bin
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import time
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d

st = time.time()

# frame settings
fps = 20
init_frame = 0
Nf = 5500

# animation file name
anim_file_name = "sumFFT_allProf1Frame_animation"

# save settings
save_mp4 = True
save_gif = False

# paths
seq_path = 'Z:\\wave_pool\\10272021E_bc\\'
bin_path = seq_path + 'temporal_nooffset\\Phase\\Float\\Bin\\'

# animation file path
anim_file = seq_path + anim_file_name

def FramesFFT(frame_number):
    yf = []
    frame_num_str = str(init_frame + frame_number).zfill(5)
    bin_file = 'Z:\\wave_pool\\10272021A_bc\\temporal_nooffset\\Phase\\Float\\Bin\\' + frame_num_str + "_phase.bin"
    xa, ya, z, header = read_bin(bin_file)
    # convert to microns from meters
    xa = xa*10**6
    ya = ya*10**6
    z = z*10**6
    # loop through all lines of a frame
    # does guassian smoothing, does FFT, and adds the data to the collection of FFT results
    for line in range(800):
        y = z[line][0:800]
        y = gaussian_filter1d(y,3)
        yf.append(np.abs(rfft(y)))
    # add all FFT results together to get a single dataset representing this DHM measurement
    sumYF = sum(yf)/800
    plot[0].remove()
    txt[0].remove()
    plot[0], = ax.plot(xf, np.log(sumYF), color='blue')
    txt[0] = ax.text(0.12,7,"frame: " + frame_num_str)

fig = plt.figure(figsize=(6,4), dpi=200, facecolor='gray', linewidth=2)
#ax1 = fig.add_subplot(2,1,1)
#ax1.set(xlabel='wavenumber (???log)', ylabel='amplitude (???log)', title='Cumulative spatial FFT')
ax = fig.add_subplot(1,1,1, xlim=(0,0.16), ylim=(0,10))
# ax.set(xlabel='wavenumber (???)', ylabel='amplitude (???log)')

frame_num_str = str(init_frame).zfill(5)
bin_file = bin_path + frame_num_str + "_phase.bin"
xa, ya, z, header = read_bin(bin_file)
# convert to microns from meters
xa = xa*10**6
ya = ya*10**6
z = z*10**6
# set the x-axis for the FFT; wave numbers in this case
N = len(xa)
D = np.amax(xa)
xf = rfftfreq(N,D/N)
yf = []

for line in range(800):
        y = z[line][0:800]
        y = gaussian_filter1d(y,3)
        yf.append(np.abs(rfft(y)))
# add all FFT results together to get a single dataset representing this DHM measurement
sumYF = sum(yf)/800

#plot1 = [ax1.plot(np.log(xf), np.log(sumYF))]
plot, = [ax.plot(xf, np.log(sumYF))]
txt = [ax.text(0.12,7,"frame: " + frame_num_str)]

# run animation
anim = animation.FuncAnimation(fig, FramesFFT, Nf, interval=50)

# save animation
if save_mp4:
    anim.save(anim_file + '.mp4',writer='ffmpeg',fps=fps)
if save_gif:
    anim.save(anim_file + '.gif',writer='ffmpeg',fps=fps)

print(time.time()-st)