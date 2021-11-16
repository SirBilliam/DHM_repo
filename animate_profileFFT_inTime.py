# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:44:31 2021

plot a profile and corresponding spatial FFT, then animate it through time
    (plot each frame in turn)

@author: bill
"""

from access_data import get_sequence, get_single_frame
import matplotlib.pyplot as plt
import numpy as np
import time
import platform
import os
from matplotlib.animation import FuncAnimation
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d


# sequence id
seq_id = '10122021A_bc'
exp_type = 'wave_pool'
data_file_postfix = '_cal_test'
anim_file_postfix = 'profileFFT_animation'
save_folder = 'media'

# sequence to extract (cutoff: 'time' or 'frame')
cutoff = 'frame'
init_time = 1.5
final_time = 2.5
init_frame = 0
final_frame = 10000
fps = 15
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

# get time and frame vectors
t, frames = get_sequence(seq_id,exp_type,cutoff=cutoff,init_time=init_time,final_time=final_time, \
                         init_frame=init_frame,final_frame=final_frame)

# animation file path
anim_file_name = seq_id[0:10] + anim_file_postfix
anim_file = seq_dir + save_folder + cd + anim_file_name

# initialize data and figure
x, y, z = get_single_frame(seq_id, exp_type, frames[0], smoothing=smoothing, \
                           sigma_smooth=sigma,file_postfix=data_file_postfix, get_xy=True)
# take the middle profile
Y = z[399]
X = x
Y = gaussian_filter1d(Y,sigma)
yf = np.abs(rfft(Y))
N = len(X)
D = np.amax(X)
xf = rfftfreq(N,D/N)

#initialize the figure window, axis, and plot element
fig = plt.figure(figsize=(6,8), dpi=200, facecolor='gray', linewidth=2)
ax1 = fig.add_subplot(2,1,1)
ax1.set(xlabel='horizontal dimension (microns)', ylabel='height (microns)',
       title='profile from DHM')
Y_mid = Y[len(X)//2]
ax1.set_ylim(Y_mid-150,Y_mid+150)

ax2 = fig.add_subplot(2,1,2, xlim=(0.00,0.16), ylim=(-2,12))
ax2.set(xlabel='wavenumber (?)', ylabel='amplitude (??)',
       title='FFT on DHM profile')

plot1, = [ax1.plot(X, Y)]
plot2, = [ax2.plot(xf, np.log(np.abs(yf)))]
txt = [ax1.text(1000,70,"frame: " + str(frames[0]).zfill(5))]


# create a function that generates a profile plot for each frame in the
# desired animation
def AniBinProfile(k):
    # read the frame data from hdf5 using access_data module
    _, _, z = get_single_frame(seq_id, exp_type, frames[k], smoothing=smoothing, \
                               sigma_smooth=sigma,file_postfix=data_file_postfix, get_xy=True)
    # take the middle profile
    Y = z[399]
    Y = gaussian_filter1d(Y,sigma)
    yf = np.abs(rfft(Y))

    #update plot
    plot1[0].remove()
    plot2[0].remove()
    txt[0].remove()
    plot1[0], = ax1.plot(X, Y, color='blue')
    plot2[0], = ax2.plot(xf, np.log(np.abs(yf)), color='green')
    txt[0] = ax1.text(1000,70,"frame: " + str(frames[k]).zfill(5))
    Y_mid = Y[len(X)//2]
    ax1.set_ylim(Y_mid-150,Y_mid+150)

def check_create_dir(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

#create animation
anim = FuncAnimation(fig, AniBinProfile, len(frames), interval=50)

# save animation
check_create_dir(seq_dir + save_folder)
anim.save(anim_file + '.mp4', fps=fps, extra_args=['-vcodec','libx264'])