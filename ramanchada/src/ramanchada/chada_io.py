# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Feb 26, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# Python libraries
import numpy as np
import os
import h5py
import time
import re
import zipfile
# Third party spectrum readers 
from renishawWiRE import WDFReader
from specio import specread


def create(source_path, target_path = '', transformers = []):
    # 1.	Create CHADA file archive and include a copy of the Native Data file
    filename, file_extension = os.path.splitext(source_path)
    if target_path == '': target_path = filename + ".cha"
    # Choose matching native file format reader according to filename extension
    # (.spc, .wdf, .txt, .csv, …), or user specification.
    reader = getReader(file_extension)
    # Import Native file using the matching reader included in the CHARISMA software.
    try:
        x_data, y_data, metadata = reader(source_path)
    except Exception as err:
        raise err
    # Get rid of bytes that are found in some of the formats
    metadata = cleanMeta(metadata)
    # Flatten metadata
    metadata = dict(zip(metadata.keys(), [str(v) for v in metadata.values()]))
    # Extract metadata from native metadata and spectrum data, store in metadata dictionary, and include in CHADA archive.
    metadata["Generated on"] = time.ctime()
    metadata["Original file"] = os.path.basename(source_path)
    # Check if list of initial transformations has been given by user (e.g. “-b –s –c[310,1890]“ = baseline + smooth + crop Raman Shifts to 310 – 1,890 1/cm).
    # ...
    #dynamic_metadata = dynamicMetaDataUpdate(x_data, y_data)
    commits = ["Generated CHADA on " + time.ctime()]
    # Make HDF5 file
    f = h5py.File(target_path, "w")
    # Store Raman dataset + label
    xy = f.create_dataset("Raman data", data=np.vstack((x_data, y_data)))
    xy.dims[0].label = 'Raman shift [1/cm]'
    xy.dims[1].label = 'Counts'
    # Store metadata
    xy.attrs.update(metadata)
    #for key in static_metadata: xy.attrs[key] = static_metadata[key]
    #for key in dynamic_metadata: xy.attrs[key] = dynamic_metadata[key]
    t = [str(tr) for tr in transformers]
    dt = h5py.special_dtype(vlen=str)
    f.create_dataset("Transformers", data=t, dtype=dt)
    c = [str(co) for co in commits]
    dt = h5py.special_dtype(vlen=str)
    f.create_dataset("Commits", data=c, dtype=dt)
    f.close()
    return target_path

def cleanMeta(meta):
    # This cleans complex-strcutures metadata, and returns a dict
    if type(meta) == dict:
        meta = {i:meta[i] for i in meta if i!=""}
        for key, value in meta.items():
            meta[key] = cleanMeta(value)
    if type(meta) == list:
        for ii, value in enumerate(meta):
            meta[ii] = cleanMeta(value)
    if type(meta) == str:
        meta = meta.replace('\\x00', '')
        meta = meta.replace('\x00', '')
    if type(meta) == bytes:
        try:
            meta = meta.decode('utf-8')
            meta = cleanMeta(meta)
        except: meta = []
    return meta

# =================3rd party file import=====================================
    
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

# =================Text and CSV import=====================================

def readTXT(file, x_col=0, y_col=0, verbose=True):
    msg= ""
    # open .txt and read as lines
    d = open(file)
    lines = d.readlines()
    # Find data lines and convert to np.array
    start, stop = startStop(lines)
    msg += "Importing " + str(stop-start+1) + " data lines starting from line " + str(start) + " in " + os.path.basename(file) + ".\n"
    data_lines = lines[start:stop]
    data = dataFromTxtLines(data_lines)
    # if columns not specified, assign x (Raman shift) and y (counts) axes
    if x_col == y_col == 0:
        # x axis is the one with mean closest to 1750
        score = 1./np.abs(data.mean(0)-1750)
        # x axis must be monotonous!
        s = np.sign(np.diff(data, axis=0))
        mono = np.array([np.all(c == c[0]) for c in s.T]) * 1.
        score *= mono
        x_col = np.argmax(score)
        # y axis is the one with maximal std/mean
        score = np.nan_to_num(data.std(0)/data.mean(0), nan=0)
        # Do not choose x axis again for y
        score[x_col] = -1000
        y_col = np.argmax(score)
        # if there's mroe than 2 columns and a header line
        if startStop(lines)[0] > 0 and data.shape[1]>2:
            msg += "Found more than 2 data columns in " + os.path.basename(file) + ".\n"
            header_line = lines[startStop(lines)[0]-1].strip('\n')
            header_line = [s.casefold() for s in re.split(';|,|\t', header_line)]
            # x axis is header line with "Shift"
            indices = [i for i, s in enumerate(header_line) if 'shift' in s]
            if indices != []:
                x_col = indices[0]
                msg += "X data: assigning column labelled '" + header_line[x_col] + "'.\n"
            else: msg += "X data: assigning column # " + str(x_col) + ".\n"
            # y axis is header line with "Subtracted"
            indices = [i for i, s in enumerate(header_line) if 'subtracted' in s]
            if indices != []:
                y_col = indices[0]
                msg += "Y data: assigning column labelled '" + header_line[y_col] + "'.\n"
            else: msg += "Y data: assigning column # " + str(y_col) + ".\n"
    x, y = data[:,x_col], data[:,y_col]
    # is x inverted?
    if all(np.diff(x) <= 0):
        x = np.flip(x)
        y = np.flip(y)
    meta_lines = lines[:startStop(lines)[0]]
    msg += "Importing " + str(start) + " metadata lines from " + os.path.basename(file) + ".\n"
    meta_lines = [re.split(';|,|\t|=', l.strip()) for l in meta_lines]
    ml = {}
    for l in meta_lines: ml.update({l[0]: l[1:]})
    # is x axis pixel numbers instead of Raman shifts?
    if all(np.diff(x) == 1) and (x[0] == 0 or x[0] == 1):
        if "Start WN" in ml:
            start_x = np.int(np.array(ml["Start WN"])[0])
        if "End WN" in ml:
            stop_x = np.int(np.array(ml["End WN"])[0])
        x = np.linspace(start_x, stop_x, len(x))
        msg += "X data: using linspace from " + str(start_x) + " to " + str(stop_x) + " 1/cm.\n"
    if verbose: print(msg)
    return x, y, ml

def dataFromTxtLines(data_lines):
    data = []
    for ii, l in enumerate(data_lines):
        l = l.strip('\n').replace("\t", " ")
        # if line has ";", it is the separator
        if ";" in l: separator = ";"
        # if line has "," AND ".", then "," is the separator
        elif "," in l and "." in l: separator = ","
        # if line has "," AND " ", then " " is the separator
        elif "," in l and " " in l: separator = " "
        # if line has only ",", then "," is the separator
        elif "," in l: separator = ","
        # or else it is " "
        else: separator = " "
        items = l.split(separator)
        # convert to float
        items = [item.replace(",", ".") for item in items]
        items = [item.replace(" ", "0") for item in items]
        items = [float(item) for item in items if item != ""]
        data.append(items)
    return np.array(data)
    
def isDataLine(line):
    line = line.strip("\n").replace("\t", " ")
    # is not blank
    #length = len(line) > 3
    blank = all([c == " " for c in line])
    # has more than 75% digits
    digits = np.sum( [d.isdigit() for d in line] ) / len(line) > .25
    # apart from digits, has only ".", ";", ",", " "           
    chars = all([c in '.,;+-eE ' for c in line if not c.isdigit()])
    return (not blank) & digits & chars

def startStop(lines):
    start_line, stop_line = 0, 0
    for ii, line in enumerate(lines):
        # if this is a data line and the following 5 lines are also data lines, then here is the start line
        if (len(lines) - ii) > 5 and start_line == 0:
            if all([isDataLine(l) for l in lines[ii:ii+5]]): start_line = ii
        # if this is a data line and the following 5 lines are also data lines, then here is the start line
        if (not isDataLine(line)) and stop_line <= start_line:
            stop_line = ii
    if stop_line <= start_line:
        stop_line = len(lines)-1
    return start_line, stop_line
    
def getYDataType(y_data):
    types = {0: "Single spectrum", 1: "Line scan", 2: "Map", 3: "Map series / volume"}
    return types[len(y_data.shape)-1]

def getReader(file_extension):
    readers = {'.spc': readSPC, '.wdf': readWDF, '.txt': readTXT, '.txtr': readTXT, '.csv': readTXT}
    return readers[file_extension]


def createZIP(source_path, target_path = '', transformers = []):
    # 1.	Create CHADA file archive and include a copy of the Native Data file
    filename, file_extension = os.path.splitext(source_path)
    if target_path == '': target_path = filename + ".cha"
    # Choose matching native file format reader according to filename extension
    # (.spc, .wdf, .txt, .csv, …), or user specification.
    reader = getReader(file_extension)
    # Import Native file using the matching reader included in the CHARISMA software.
    try:
        x_data, y_data, static_metadata = reader(source_path)
    except Exception as err:
        raise err
    # Extract metadata from native metadata and spectrum data, store in metadata dictionary, and include in CHADA archive.
    static_metadata["Generated on"] = time.ctime()
    static_metadata["Original file"] = os.path.basename(source_path)
    # Check if list of initial transformations has been given by user (e.g. “-b –s –c[310,1890]“ = baseline + smooth + crop Raman Shifts to 310 – 1,890 1/cm).
    #dynamic_metadata = dynamicMetaDataUpdate(x_data, y_data)
    commits = ["Generated CHADA on " + time.ctime()]
    zf = zipfile.ZipFile(target_path, mode="w", compression=zipfile.ZIP_DEFLATED)
    zf.write(source_path, os.path.basename(source_path))
    zf.writestr("static_meta.txt", str(static_metadata))
    #zf.writestr("dynamic_meta.txt", str(dynamic_metadata))
    zf.writestr("transformers.txt", str(transformers))
    zf.writestr("commits.txt", str(commits))
    zf.close()
    return