"""
manipulate koala bin files
"""

import numpy as np

def read_bin(fname):
    """
    read a koala bin file and return surface data, header dictionary
    """

    # define data types
    bin_header_dtype = np.dtype([
     ("version", "u1"), 
     ("endian", "u1"),
     ("head_size", "i4"),
     ("width", "i4"),           # pixel width
     ("height", "i4"),          # pixel height
     ("px_size", "f4"),         # pixel size [m]    
     ("hconv", "f4"),           # height conversion (-> m)
     ("unit_code", "u1")        # 1=rad 2=m
    ])

    # read file
    f = open(fname, 'rb')
    bin_header = np.fromfile(f, dtype=bin_header_dtype, count=1)
    shape = ((int)(bin_header['height']), (int)(bin_header['width']))
    z_data = np.fromfile(f, dtype='float32').reshape(shape)*bin_header["hconv"]
    f.close()

    # axes
    x_axis = np.arange(bin_header["width"])*bin_header["px_size"]
    y_axis = np.arange(bin_header["height"])*bin_header["px_size"]

    return (x_axis, y_axis, z_data, bin_header)
