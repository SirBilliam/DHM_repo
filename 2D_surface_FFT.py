# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:25:37 2021

@author: bill
"""

import scipy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.signal as sig
import binkoala

phase, header = binkoala.read_mat_bin("Z:\\wave_pool\\10122021A_bc\\Phase\\Float\\Bin\\01500_phase.bin")
plt.pcolormesh(phase, cmap='Greys')
plt.show()

# z = fft.fftn(phase)
# y = np.linspace(0, 1255, num=800)
# x = np.linspace(0, 1255, num=800)
# plt.pcolormesh(np.real(z)/np.amax(np.real(z)), cmap=cm.Reds)
# plt.show()

from scipy import fftpack
im_fft = fftpack.fft2(phase)

# Show the results

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

plt.figure()
plot_spectrum(im_fft)
plt.title('Fourier transform')