# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:02:40 2021

@author: bill
"""

import numpy as np
import matplotlib.pyplot as plt


# First create some toy data:
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# Create a figure
fig = plt.figure()

# # Create a subplot
# ax = fig.subplots()
# ax.plot(x, y)
# ax.set_title('Simple plot')

# # Create two subplots and unpack the output array immediately
# ax1, ax2 = fig.subplots(1, 2, sharey=True)
# ax1.plot(x, y)
# ax1.set_title('Sharing Y axis')
# ax2.scatter(x, y)

# # Create four polar Axes and access them through the returned array
# axes = fig.subplots(2, 2, subplot_kw=dict(projection='polar'))
# axes[0, 0].plot(x, y)
# axes[1, 1].scatter(x, y)

# # Share a X axis with each column of subplots
# fig.subplots(2, 2, sharex='col')

# # Share a Y axis with each row of subplots
# fig.subplots(2, 2, sharey='row')

# Share both X and Y axes with all subplots
fig.subplots(2, 2, sharex='all', sharey='all')

# # Note that this is the same as
# fig.subplots(2, 2, sharex=True, sharey=True)