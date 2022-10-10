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
import os
import h5py,h5pyd
import time
import zipfile
from datetime import datetime
# ramanchada imports
from ramanchada.file_io.third_party_readers import readSPC, readWDF, readOPUS
from ramanchada.file_io.txt_format_readers import read_JCAMP, readTXT
from ramanchada.file_io.binary_readers import readSPA, read_ngs


def import_native(source_path):
    # 1.	Create CHADA file archive and include a copy of the Native Data file
    filename, file_extension = os.path.splitext(source_path)
    # Choose matching native file format reader according to filename extension
    # (.spc, .wdf, .txt, .csv, …), or user specification.
    reader = getReader(file_extension.lower())
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
    metadata["Original file"] = os.path.basename(source_path)
    return x_data, y_data, metadata

def getReader(file_extension):
    readers = {'.spc': readSPC, '.sp': readSPC,
               '.spa': readSPA,
               '.0': readOPUS, '.1': readOPUS, '.2': readOPUS,
               '.wdf': readWDF,
               '.ngs': read_ngs,
               '.jdx': read_JCAMP, '.dx': read_JCAMP,
               '.txt': readTXT,'.txtr': readTXT, '.csv': readTXT, '.prn': readTXT, '.rruf': readTXT}
    return readers.get(file_extension, readTXT)

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


def read_chada(file_path, raw=False, h5module=h5py):
    # Open HDF5
    with h5module.File(file_path, "r") as f:
        keys = list(f.keys())
        if raw:
            dset = f['raw']
        elif len(keys) < 2:
            dset = f[keys[0]]
        else:
            # load the one that is not 'raw'
            right_key = [key for key in keys if (key != 'raw') and (isinstance(f[key], h5module.Dataset))][0]
            dset = f[right_key ]
        # Load metadata
        a = dset.attrs
        meta = dict(zip(a.keys(), [str(v) for v in a.values()]))
        # Load xy data
        x, y = dset[0], dset[1]
        # Load data labels
        x_label, y_label = dset.dims[0].label, dset.dims[1].label
    return x, y, meta, x_label, y_label

def get_chada_commits(file_path, commit=[],h5module=h5py):
    # Open HDF5
    #with h5py.File(file_path, "r", track_order=True) as f:
    with h5module.File(file_path, "r",track_order=True) as f:        
        commits = list(f.keys())
    return commits

def write_chada(file_path, dset_name, x, y, metadata, mode='a', x_label = 'Raman shift [1/cm]', y_label = 'raw counts [1]',h5module=h5py):
    # Create HDF5 file
    with h5module.File(file_path, mode) as f:
        # Store Raman dataset + label
        xy = f.create_dataset(dset_name, data=np.vstack((x, y)))
        xy.dims[0].label = x_label
        xy.dims[1].label = y_label
        # Store metadata
        xy.attrs.update(metadata)
    
def write_new_chada(file_path, x, y, metadata,h5module=h5py):
    metadata["CHADA generated on"] = time.ctime()
    write_chada(file_path, "raw", x, y, metadata, mode='w',h5module=h5py)
    
def create_chada_from_native(native_filename, chada_filename=[],h5module=h5py):
    if chada_filename == []:
        name, _ = os.path.splitext(native_filename)
        chada_filename = name + ".cha"
    x, y, meta = import_native(native_filename)
    write_new_chada(chada_filename, x, y, meta,h5module)
    return chada_filename
    
def commit_chada(spectrum, commit_text, append=False,h5module=h5py):
    if commit_text == 'raw':
        print('Raw cannot be edited!')
        return
    if not append:
        with h5module.File(spectrum.file_path, "a") as f:
            for key in list(f.keys()):
                if key != 'raw':
                    del f[key]
    write_chada(spectrum.file_path, commit_text, spectrum.x, spectrum.y, spectrum.meta,\
        x_label=spectrum.x_label, y_label=spectrum.y_label,h5module=h5module)
