# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Feb 26, 2021
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
# Third party spectrum readers 
from renishawWiRE import WDFReader
from specio import specread
import opusFC
#import spectrochempy as scp
    
def readWDF(file):
    s = WDFReader(file)
    y_data = s.spectra
    x_data = s.xdata
    if np.mean(np.diff(x_data)) < 0:
        y_data = np.flip(y_data)
        x_data = np.flip(x_data)
    static_metadata = {
        "laser wavelength": s.laser_length,
        "no. of accumulations": s.accumulation_count,
        "spectral unit": s.spectral_unit.name,
        "OEM software name": s.application_name,
        "OEM software version": s.application_version
        }
    return x_data, y_data, static_metadata

def readSPC(file):
    s = specread(file)
    y_data = s.amplitudes
    x_data = s.wavelength
    if np.mean(np.diff(x_data)) < 0:
        y_data = np.flip(y_data)
        x_data = np.flip(x_data)
    static_metadata = s.meta
    return x_data, y_data, static_metadata

def readOPUS(file, obj_no=0):
    c = opusFC.listContents(file)
    data = opusFC.getOpusData(file, c[obj_no])
    return data.x, data.y, data.parameters

# def readSPA(file):
#     s = scp.read(file)
#     m = s.description.split('\n')
#     meta_dict = dict([mm.strip('##').split(':') for mm in m])
#     return np.array(s.x.values), s.data[0], meta_dict