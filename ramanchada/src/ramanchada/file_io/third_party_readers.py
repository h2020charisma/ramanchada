# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Feb 26, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

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