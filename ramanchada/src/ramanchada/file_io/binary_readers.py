# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Sept 15, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# Python libraries
import numpy as np
import struct

def readSPA(filename, flip = True):
    """function to read k-vector and spectrum from a *.SPA file
    
    :param filename: full path to the file to be read
    :type filename: str
    :return: k-vector and spectrum as separate arrays 
    :rtype: (np.array, np.array)
    """
    k = None
    spec = None
    with open(filename, 'rb') as f:    
        # the data offset is saved at offset 386
        f.seek(386, 0)
        offset = struct.unpack("i", f.read(4))[0]
        # the number of data points is saved at offset 564
        f.seek(564, 0)
        n = struct.unpack("i", f.read(4))[0]
        # the max and min wavenumbers are saved at 576 and 580 respectively
        f.seek(576, 0)
        w_max = struct.unpack("f", f.read(4))[0]
        w_min = struct.unpack("f", f.read(4))[0]
        k = np.linspace(w_min, w_max, n)
        # read the data points
        f.seek(offset, 0)
        spec = np.array([struct.unpack("f", f.read(4))[0] for i in range(n)])
        if flip:
            spec = np.flip(spec)
        # set corrupt counts to zero
        spec[spec>1e5] = 0
    return k, spec, {}