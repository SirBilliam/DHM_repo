# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:56:36 2021

fit the zero frame to 2nd order surface and subtract that from a test frame
then plot the calibrated 3D surface

@author: bill
"""
from access_data import get_single_frame
import calibrate_surface as cs
import numpy as np
import matplotlib.pyplot as plt
from binkoala2 import read_bin
import time


seq_id = '10272021A_bc'
cal_frame = 0
test_frame = 1100
x_range = range(800)
y_range = range(800)

Zero = get_single_frame(seq_id, cal_frame, x_range, y_range)

Z_cal, a, b = cs.fit_surface(Zero)

Z = get_single_frame(seq_id, test_frame, x_range, y_range)

Z_out = cs.calibrate_surface(Z, Z_cal)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-50,50)

x_axis, y_axis, z, bin_header = read_bin(r"Z:\wave_pool\10272021A_bc\temporal_nooffset\Phase\Float\Bin\00000_phase.bin")

X, Y = np.meshgrid(x_axis*10**6,y_axis*10**6)

ax.plot_surface(X, Y, Z_out*10**6, cmap="magma")
