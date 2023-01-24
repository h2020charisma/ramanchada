# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Sept 15, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# MIT License
#
# Copyright (c) 2021–2022 CHARISMA H2020 project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Python libraries
import numpy as np
import struct
import matplotlib.pyplot as plt
from itertools import islice

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

def read_ngs(file, show=True):
    with open(file, "rb") as f:
        # read NextGen string
        s = f.read(10)
        # if the string does not match, abort
        nextgen_string = s.decode("utf-8").lower()
        if nextgen_string != 'ngsnextgen':
            print('Not a readable file !')
            return
        # read DataMatrix string form byte #18
        s = f.seek(18)
        # read length of string as single byte
        s = f.read(1)
        l = int.from_bytes(s, "big") 
        # read the actual string
        s = f.read(l)
        datamatrix_string =  s.decode("utf-8").lower()
        # if the string does not match, abort
        if datamatrix_string != 'datamatrix':
            print('Not a readable file !')
            return
        # read filename at byte #38
        s = f.seek(38)
        # read length of string as single byte
        s = f.read(1)
        l = int.from_bytes(s, "big")
        # read the actual string
        s = f.read(l)
        file_name = s.decode("utf-8")
        # Read no. of channels as 32 bit integer, 16 bytes from end of filename
        s = f.seek(16, 1)
        s = f.read(4)
        n = struct.unpack('i', s)[0]
        print(f'Reading Labspec .ngs file {file_name} with {n} channels.')
        # Read data block, starting 8 bytes from end of channel num. Each y count is a 32 bit integer.
        y = read_4byte_datablock(f, n, 8)
        # Read parameter block. Before, there's a rather complicated sequence of skipping obsolete parameters...
        f.seek(2, 1)
        s = f.read(1)
        l = int.from_bytes(s, "big")
        # Skip bytes as long as they are zeros
        if l == 0:
            while l == 0:
                s = f.read(1)
                l = int.from_bytes(s, "big")
            f.seek(1, 1)
            s = f.read(1)
            l = int.from_bytes(s, "big")
        #
        f.seek(l, 1)
        s = f.read(1)
        l = int.from_bytes(s, "big")
        #
        f.seek(l, 1)
        f.seek(2, 1)
        s = f.read(1)
        l = int.from_bytes(s, "big")
        #
        f.seek(l, 1)
        s = f.read(1)
        l = int.from_bytes(s, "big")
        #
        f.seek(l, 1)
        f.seek(16, 1)
        # Finally, read the start and end wavenumbers (start_x)
        s = f.read(4)
        start_x = struct.unpack('f', s)[0]
        s = f.read(4)
        end_x = struct.unpack('f', s)[0]
        # Check whether end_x is equal to no. of channels
        if end_x == n:
            x = read_4byte_datablock(f, n, 20)
        else:
            # Construct x axis
            x = np.linspace(start_x, end_x, n)
        meta = read_ngs_meta(f)

        # plot
    if show:
        plt.figure()
        plt.plot(x,y)
    return x, y, meta

def read_4byte_datablock(f, length, skip=8):
    s = f.seek(skip, 1)
    y = np.zeros(length)
    for ii in range(length):
        s = f.read(4)
        y[ii] = struct.unpack('f', s)[0]
    return y

def read_bytestring(f):
    # read length of string as single byte
    s = f.read(1)
    l = int.from_bytes(s, "big") 
    # read the actual string
    s = f.read(l)
    return s.decode('iso-8859-1')

def read_ngs_meta(f):
    position = f.tell()+1
    f.seek(position)
    abl = ''
    while abl != 'Table':          #zuerst: Table Param
        s = f.read(1)
        l = int.from_bytes(s, "big")
        if l == 5:
            f.seek(-1, 1)
            abl = read_bytestring(f)
        else:
            position += 1
            f.seek(position)
    abl = ''
    position = f.tell()
    f.seek(position)
    while abl != 'Table':         
        s = f.read(1)
        l = int.from_bytes(s, "big")
        if l == 5:
            f.seek(-1, 1)
            abl = read_bytestring(f)
            position = f.tell()
        else:
            position += 1
            f.seek(position)
    abl = read_bytestring(f)
    # Make dictionary
    meta = {}
    if abl.upper() == 'ACQ':
        abl = read_bytestring(f)
        f.seek(4, 1)
        # read no. of params
        s = f.read(1)
        l = int.from_bytes(s, "big")
        f.seek(1, 1)
        par_names = []
        # Read parameter names
        for ii in range(l):
            abl = read_bytestring(f)
            par_names.append(abl)
        # Skip some stuff
        f.seek(2, 1)
        abl = read_bytestring(f)
        abl = read_bytestring(f)
        f.seek(10, 1)
        s = f.read(1)
        # read no. of values
        l = int.from_bytes(s, "big")
        f.seek(1, 1)
        # Read parameter values
        par_values = []
        for ii in range(l):
            abl = read_bytestring(f)
            par_values.append(abl)
        f.seek(2, 1)
        # Read parameter units
        par_units = []
        for ii in range(l):
            abl = read_bytestring(f)
            # add unit to dict key and exchange for new key
            par_units.append(abl)
        # add units to par names
        par = [ name + f' [{unit}]' for name, unit in zip(par_names, par_units) ]
        # Construct meta dict
        meta = dict( zip(par, par_values) )
        return meta