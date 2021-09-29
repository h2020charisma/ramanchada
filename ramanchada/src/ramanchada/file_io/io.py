# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Feb 26, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# Python libraries
import numpy as np
import os
import h5py
import time
import zipfile
from datetime import datetime
# ramanchada imports
from file_io.third_party_readers import readSPC, readWDF, readOPUS
from file_io.txt_format_readers import read_JCAMP, readTXT
from file_io.binary_readers import readSPA


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

def read_chada(file_path, raw=False):
    # Open HDF5
    with h5py.File(file_path, "r") as f:
        if raw:
            dset = f['raw']
        else:
            dset = f[ list(f.keys())[-1] ]
        # Load metadata
        a = dset.attrs
        meta = dict(zip(a.keys(), [str(v) for v in a.values()]))
        # Load xy data
        x, y = dset[0], dset[1]
        # Load data labels
        x_label, y_label = dset.dims[0].label, dset.dims[1].label
    return x, y, meta, x_label, y_label

def get_chada_commits(file_path, commit=[]):
    # Open HDF5
    with h5py.File(file_path, "r", track_order=True) as f:
        commits = list(f.keys())
    return commits

def write_chada(file_path, dset_name, x, y, metadata, x_label = 'Raman shift [1/cm]', y_label = 'raw counts [1]'):
    # Create HDF5 file
    with h5py.File(file_path, "a", track_order=True) as f:
        # Store Raman dataset + label
        xy = f.create_dataset(dset_name, data=np.vstack((x, y)), track_order=True)
        xy.dims[0].label = x_label
        xy.dims[1].label = y_label
        # Store metadata
        xy.attrs.update(metadata)
    
def write_new_chada(file_path, x, y, metadata):
    metadata["CHADA generated on"] = time.ctime()
    write_chada(file_path, "raw", x, y, metadata)
    
def create_chada_from_native(native_filename, chada_filename=[]):
    if chada_filename == []:
        name, _ = os.path.splitext(native_filename)
        chada_filename = name + ".cha"
    x, y, meta = import_native(native_filename)
    write_new_chada(chada_filename, x, y, meta)
    return chada_filename
    
def commit_chada(spectrum, commit_text, append=False):
    if commit_text == 'raw':
        print('Raw cannot be edited!')
        return
    if not append:
        with h5py.File(spectrum.file_path, "a", track_order=True) as f:
            for key in list(f.keys()):
                if key != 'raw':
                    del f[key]
    write_chada(spectrum.file_path, commit_text, spectrum.x, spectrum.y, spectrum.meta, spectrum.x_label, spectrum.y_label)

def getYDataType(y_data):
    types = {0: "Single spectrum", 1: "Line scan", 2: "Map", 3: "Map series / volume"}
    return types[len(y_data.shape)-1]

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

def timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S ")