# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Feb 26, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# Python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import wiener, savgol_filter, find_peaks
from scipy import sparse
from scipy.optimize import dual_annealing
from scipy.sparse.linalg import spsolve
import os
import zipfile
import time
import ast
from scipy.interpolate import interp1d
from sklearn.decomposition import NMF
# Third party spectrum readers 
from renishawWiRE import WDFReader
from specio import specread

# 1.	Create CHADA file archive and include a copy of the Native Data file
def create(source_path, target_path = '', transformers = []):
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
    dynamic_metadata = dynamicMetaDataUpdate(x_data, y_data)
    commits = ["Generated CHADA on " + time.ctime()]
    zf = zipfile.ZipFile(target_path, mode="w", compression=zipfile.ZIP_DEFLATED)
    zf.write(source_path, os.path.basename(source_path))
    zf.writestr("static_meta.txt", str(static_metadata))
    zf.writestr("dynamic_meta.txt", str(dynamic_metadata))
    zf.writestr("transformers.txt", str(transformers))
    zf.writestr("commits.txt", str(commits))
    zf.close()
    return

def makeXCalFromSpec(target_file, reference_file, bounds=[-10.,10.]):
    T = Chada(target_file)
    R = Chada(reference_file)
    # Get peak positions from target_spectrum
    T.peaks()
    peak_pos = np.array(T.bands['peak pos [1/cm]'])
    # interpolate reference to target x_data 
    f_inter = interp1d(R.x_data, R.y_data, kind="cubic", bounds_error=False, fill_value=0)
    reference = f_inter(T.x_data)
    # Maximize HQI by simulated annealing
    align_params = [T.y_data, reference, T.x_data, peak_pos]
    lw = [bounds[0]] * len(peak_pos)
    up = [bounds[1]] * len(peak_pos)
    ret = dual_annealing( align_score, bounds=list(zip(lw, up)), args=align_params, seed=1234 )
    return peak_pos, ret.x

def align_score(shifts_at_peaks, y, y_ref, x, peak_pos):
    # extrapolate shift vector
    f_inter = interp1d(peak_pos, shifts_at_peaks, kind="cubic", bounds_error=False, fill_value="extrapolate")
    shifts = f_inter(x)
    # calculate HQI of shifted spectrum and ref
    return 1. / hqi(y_ref, spec_shift(y, x, shifts))

def spec_shift(y0, x0, shifts, show=False):
    f_inter = interp1d(x0+shifts, y0, kind="cubic", bounds_error=False, fill_value=0)
    y_shifted = f_inter(x0)
    if show:
        plt.figure()
        plt.plot(x0, y0, label="original")
        plt.plot(x0, y_shifted, label="Shifted")
        plt.legend()
    return y_shifted

def hqi(y1, y2):
    return np.linalg.norm(np.dot(y1, y2))**2 / np.linalg.norm(y1)**2 / np.linalg.norm(y2)**2
    
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

def dynamicMetaDataUpdate(x_data, y_data):
    dynamic_metadata = {
        "Raman data type": getYDataType(y_data),
        "xy dimensions":  y_data.shape[1:],
        "no. of channels": y_data.shape[0],
        "minimum wavelength": x_data.min(),
        "maximum wavelength": x_data.max(),
        "mean counts": y_data.mean(),
        "standard deviation": y_data.std(),  
        }
    return dynamic_metadata
    
def getYDataType(y_data):
    types = {0: "Single spectrum", 1: "Line scan", 2: "Map", 3: "Map series / volume"}
    return types[len(y_data.shape)-1]

def getReader(file_extension):
    readers = {'.spc': readSPC, '.wdf': readWDF}
    return readers[file_extension]

def lims(Y, x, x_min, x_max):
    y_min, y_max = np.argmin(np.abs(x-x_min)), np.argmin(np.abs(x-x_max))
    return Y[...,y_min:y_max]


# =========================CLASSES===========================================
class Chada():
    def __init__(self, chada_path):
        self.path = chada_path
        try:
            # Open archive
            zf = zipfile.ZipFile(chada_path)
            # Load data
            self.static_metadata = ast.literal_eval(zf.read("static_meta.txt").decode('utf-8'))
            self.dynamic_metadata = ast.literal_eval(zf.read("dynamic_meta.txt").decode('utf-8'))
            # self.transformers will be populated upon transformers execution
            transformers = ast.literal_eval(zf.read("transformers.txt").decode('utf-8'))
            self.commits = ast.literal_eval(zf.read("commits.txt").decode('utf-8'))
            # Get native data file name from static_metadata
            # Temporarily extract native data file
            extract_path = zf.extract(self.static_metadata["Original file"], os.path.dirname(chada_path))
            # Import Native file as data using the matching reader included in the CHARISMA software.
            _, file_extension = os.path.splitext(extract_path)
            reader = getReader(file_extension)
            self.x_data_0, self.y_data_0, _ = reader(extract_path)
            self.x_data, self.y_data = self.x_data_0.copy(), self.y_data_0.copy()
            # Remove temp. file
            os.remove(extract_path)
        except Exception as err:
            raise err
        finally:    
            zf.close()
        # self.transformers will be populated upon transformers execution
        self.transformers = []
        for t in transformers:
            # t[0] has transformer function name
            transformer_func = getattr(self, t[0])
            if len(t) > 1: transformer_func(*t[1:])
            else: transformer_func()
        return
    
    def commit(self, commit = "Updated CHADA on " + time.ctime() ):
        self.commits.append(commit)
        zf = zipfile.ZipFile(self.path, "a")
        zf.writestr("dynamic_meta.txt", str(self.dynamic_metadata))
        zf.writestr("transformers.txt", str(self.transformers))
        zf.writestr("commits.txt", str(self.commits))
        zf.close()
        return
    
    def rewind(self, step):
        # Reset data
        self.x_data, self.y_data = self.x_data_0.copy(), self.y_data_0.copy()
        # Truncate transformers list
        transformers = self.transformers[:step]
        self.transformers = []
        # self.transformers will be populated upon transformers execution
        for t in transformers:
            # t[0] has transformer function name
            transformer_func = getattr(self, t[0])
            if len(t) > 1: transformer_func(*t[1:])
            else: transformer_func()
        # Just in case transformers list is empty
        self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
        return
    
    #--------------------------UTILITIES------------------------
    def plot(self, save_fig_name = ""):
        ylabel = "Intensity"
        fig = plt.figure(figsize=[8,4])
        plt.plot(self.x_data, self.y_data, 'k-')
        plt.ylabel(ylabel)
        plt.xlabel("Raman shift [rel. 1/cm]")
        plt.grid(axis='x', which='both', linestyle=':')
        #plt.yticks([])
        if save_fig_name == "":
            plt.show()
        else:
            fig.savefig(save_fig_name, dpi=100)
        return

    def peaks(self, make_plot = False):
        s = savgol_filter(self.y_data, 3, 2)
        s -= s.min()
        s /= s.max()
        p = find_peaks(s,
            height=.1,
            threshold=None,
            distance=2,
            prominence=.05,
            width=None,
            wlen=None,
            rel_height=0.1,
            plateau_size=None)
        P = pd.DataFrame(p[1])
        P["peak pos [1/cm]"] = self.x_data[p[0]]
        cols = P.columns.tolist()
        cols = [cols[-1], cols[0], cols[1], cols[2], cols[3]]
        P = P.reindex(columns=cols)
        P["left_bases"] = self.x_data[P["left_bases"]]
        P["right_bases"] = self.x_data[P["right_bases"]]
        self.bands = P
        if make_plot:
            ylabel = "Intensity"
            fig, ax = plt.subplots(figsize=[8,4])
            plt.plot(self.x_data, self.y_data, 'k-')
            for pos in p[0]:
                k_pos = self.x_data[pos]
                band_y = self.y_data[pos]
                ax.annotate(str(int(k_pos)), xy=(k_pos, band_y + self.y_data.max()*.01),  xycoords='data',
                            xytext=(k_pos, band_y + self.y_data.max()*.1), textcoords='data',
                            arrowprops=dict(facecolor='black', shrink=0.05, width=2,headwidth=5),
                            horizontalalignment='center', verticalalignment='bottom', size=10
                            )
            plt.ylim(-.05, self.y_data.max()+ self.y_data.max()*.2)
            plt.xlabel('Raman shift [1/cm]')
            plt.ylabel(ylabel)
            plt.grid(axis='x', which='both', linestyle=':')
            #plt.yticks([])
            plt.show()
        return
    
    # -----------TRANSFORMERS-------------------------------------------------------------
    def baseline(self, lam=5e5, p=0.001, niter=100, smooth=7, show=False):
       # After Eilers, P. H., & Boelens, H. F. (2005). Baseline correction with asymmetric least squares smoothing. Leiden University Medical Centre Report, 1(1), 5.
       # Fits & returns a background model
       b = self.y_data.copy()
       y = self.y_data.copy()
       if smooth > 0: y = wiener(y, smooth)
       L = len(y)
       D = sparse.csc_matrix(np.diff(np.eye(L), 2))
       w = np.ones(L)
       for i in range(niter):
           W = sparse.spdiags(w, 0, L, L)
           Z = W + lam * D.dot(D.transpose())
           z = spsolve(Z, w*y)
           w = p * (y > z) + (1-p) * (y < z)
       b = z
       self.transformers.append([ 'baselineT', b.tolist() ])
       self.y_data -= b
       self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
       if show:
           ylabel = "Intensity"
           plt.figure(figsize=[8,4])
           plt.plot(self.x_data, self.y_data + b, 'k')
           plt.plot(self.x_data, b, 'r')
           plt.ylabel(ylabel)
           plt.xlabel("Raman shift [rel. 1/cm]")
           plt.grid(axis='x', which='both', linestyle=':')
           #plt.yticks([])
       return
   
    def baselineT(self, b):
        b = np.array(b)
        self.y_data -= b
        self.transformers.append([ 'baselineT', b.tolist() ])
        return

    def x_crop(self, k_min=300, k_max=1800):
        self.transformers.append(['x_crop', k_min, k_max])
        self.y_data = lims(self.y_data, self.x_data, k_min, k_max)
        self.x_data = lims(self.x_data, self.x_data, k_min, k_max)
        self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
        return
    
    def normalize(self, norm_type = 'area'):
        self.transformers.append(['normalize', norm_type])
        self.y_data -= self.y_data.min(axis=0)
        self.y_data /= np.mean(self.y_data, axis=0)
        self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
        return
    
    def XCal(self, x_pos, shifts_pos, show=False):
        x_pos, shifts_pos = np.array(x_pos), np.array(shifts_pos)
        # Calulate shift vector
        f_inter = interp1d(x_pos, shifts_pos, kind="cubic", bounds_error=False, fill_value=0)
        shifts_vector = f_inter(self.x_data)
        aligned_target = spec_shift(self.y_data, self.x_data, shifts_vector)
        bounds = [shifts_vector.min(), shifts_vector.max()]
        if show:
            plt.figure(figsize=[8,4])
            plt.plot(self.x_data, shifts_vector)
            plt.plot(x_pos, shifts_pos, 'o')
            plt.ylim(bounds)
            plt.xlabel("Raman shift [rel. 1/cm]")
            plt.ylabel("wavenumber shift [rel. 1/cm]")
            plt.grid(linestyle=':')
            plt.show()
            plt.figure(figsize=[8,4])
            plt.plot(self.x_data, self.y_data, label='target')
            plt.plot(self.x_data, aligned_target, label='aligned target')
            plt.xlabel("Raman shift [rel. 1/cm]")
            plt.yticks([])
            plt.grid(axis='x', which='both', linestyle=':')
            plt.legend()
        # Return aligned target and shift vector
        self.transformers.append(['XCal', x_pos.tolist(), shifts_pos.tolist()])
        self.y_data = aligned_target
        self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
        return
    
    def shiftX(self, shifts, show=False):
        shifts = np.array(shifts)
        f_inter = interp1d(self.x_data + shifts, self.y_data, kind="cubic", bounds_error=False, fill_value=0)
        y_shifted = f_inter(self.x_data)
        if show:
            plt.figure()
            plt.plot(self.x_data, self.y_data, label="original")
            plt.plot(self.x_data, y_shifted, label="Shifted")
            plt.legend()
            plt.xlabel("Raman shift [rel. 1/cm]")
            plt.yticks([])
            plt.grid(axis='x', which='both', linestyle=':')
        self.transformers.append(['shiftX', shifts.tolist()])
        self.y_data = y_shifted
        self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
        return


class ChadaGroup():
    def __init__(self, chada_files):
        Y = []
        X = []
        x_increment_all = np.inf
        x_min_all = -np.inf
        x_max_all = np.inf
        self.labels = []
        for file in chada_files:
            G = Chada(file)
            X.append(G.x_data)
            Y.append(G.y_data)
            x_increment = np.abs(np.diff(G.x_data)).min()
            x_min, x_max = G.x_data.min(), G.x_data.max()
            x_min_all = np.max([x_min_all, x_min])
            x_max_all = np.min([x_max_all, x_max])
            x_increment_all = np.min([x_increment_all, x_increment])
            del G
            self.labels.append(os.path.basename(file))
        self.x_data = np.arange(x_min_all, x_max_all, x_increment_all)
        interpolated_Y = []
        for x, y in zip(X,Y):
            f_inter = interp1d(x, y, kind="cubic")
            interpolated_Y.append(f_inter(self.x_data))
        self.y_data = np.array(interpolated_Y)
        del X, Y, interpolated_Y
        return
    
    def plot(self, save_fig_name = ""):
        plotData(self.x_data, self.y_data, self.labels)
        return
    
    def distance(self, ref=0, show=True):
        D = []
        for y in self.y_data:
            D.append( y - self.y_data[ref,...] )
        self.y_differences = np.array(D)
        if show:
            plotData(self.x_data, self.y_differences, self.labels)
        return
    
    def normalize(self, normalization_type = 'area'):
        norms = {'area': [np.min,np.mean], 'minmax': [np.min,np.max], 'vector':[np.mean,np.std]}
        func1, func2 = norms[normalization_type]
        self.y_data -= func1(self.y_data, axis=1)[:,np.newaxis]
        self.y_data /= func2(self.y_data, axis=1)[:,np.newaxis]
        return
    
    def x_crop(self, k_min=0, k_max=4000):
        self.y_data = lims(self.y_data, self.x_data, k_min, k_max)
        self.x_data = lims(self.x_data, self.x_data, k_min, k_max)
        return
    
    def nmf(self, n_components):
        self.clf = NMF(n_components, init='nndsvda', solver='cd', max_iter=1000)
        self.scores = self.clf.fit_transform(self.y_data)
        comp_cols = np.char.add('NMF_component ',np.arange(n_components).astype(str))
        DF = pd.DataFrame(data= self.scores, columns=comp_cols)
        self.nmf_scores = pd.concat([pd.Series(self.labels, name='file'), DF], axis=1)
        print(self.nmf_scores)
        return
        
def plotData(x_data, y_data, labels, ylabel = "Intensity", save_fig_name = ""):
    fig = plt.figure(figsize=[8,4])
    for l, y in zip(labels, y_data):
        plt.plot(x_data, y, label=l)
    plt.ylabel(ylabel)
    plt.xlabel("Raman shift [rel. 1/cm]")
    plt.grid(axis='x', which='both', linestyle=':')
    plt.legend()
    #plt.yticks([])
    if save_fig_name == "":
        plt.show()
    else:
        fig.savefig(save_fig_name, dpi=100)
    return

#
## ============================OLD STUFF==================================
#class Chada():
#    def __init__(self, file):
#        try:
#            f = open(file, "rb", buffering=0)
#            self.binary_data = f.read()
#        except Exception as err:
#            raise err 
#        finally:    
#            f.close()
#        self.readers = {'.spc': specread, '.wdf': WDFReader}
#        self.filename = file[file.rfind(os.path.sep)+2:]
#        ext = file[file.rfind("."):]
#        reader = self.readers[ext]
#        s = reader(file)
##        # This is for .wdf (Renishaw) files only
#        counts = s.spectra
#        wavenumbers = s.xdata
##        counts = s.amplitudes
##        wavenumbers = s.wavelength
#        if np.mean(np.diff(wavenumbers)) < 0:
#            counts = np.flip(counts)
#            wavenumbers = np.flip(wavenumbers)
#        self.metadata = {
#        "original filepath": file,
#        "native format": ext,
#        "laser wavelength": s.laser_length,
#        "no. of accumulations": s.accumulation_count,
#        "spectral unit": s.spectral_unit.name,
#        "OEM software name": s.application_name,
#        "OEM software version": s.application_version,
#        "minimum wavelength": wavenumbers.min(),
#        "maximum wavelength": wavenumbers.max(),
#        "no. of channels": len(counts),
#        "mean counts": counts.mean(),
#        "standard deviation": counts.std(),  
#        }
#        return
#    
#    def meta(self):
#        return pd.DataFrame.from_dict(self.metadata, orient='index', columns=['value'])
#    
#    def data(self):
#        try:
#            specfile = TemporaryFile("wb", delete=False)
#            specfile.write(self.binary_data)
#            ext = self.metadata["native format"]
#        except Exception as err:
#            raise err
#        finally:    
#            specfile.close()
#        try:    
#            reader = self.readers[ext]
#            s = reader(specfile.name)
#        except Exception as err:
#            raise err
#        finally:
#            s.close()
#        os.unlink(specfile.name)
##        counts = s.amplitudes
##        wavenumbers = s.wavelength
##        if np.mean(np.diff(wavenumbers)) < 0:
##            counts = np.flip(counts)
##            wavenumbers = np.flip(wavenumbers)
#        counts = s.spectra
#        wavenumbers = s.xdata
#        if np.mean(np.diff(wavenumbers)) < 0:
#            counts = np.flip(counts)
#            wavenumbers = np.flip(wavenumbers)
#        if hasattr(self, 'background_model'):
#            counts -= self.background_model
#        return wavenumbers, counts
#
#    def plot(self, save_fig_name = ""):
#        wavenumbers, counts = self.data()
#        ylabel = self.metadata["spectral unit"]
#        if hasattr(self, 'background_model'):
#            ylabel = ylabel + ", background-corrected"
#        fig = plt.figure(figsize=[8,4])
#        plt.plot(wavenumbers, counts, 'k-')
#        plt.ylabel(ylabel)
#        plt.xlabel("Raman shift [rel. 1/cm]")
#        plt.grid(axis='x', which='both', linestyle=':')
#        plt.yticks([])
#        if save_fig_name == "":
#            plt.show()
#        else:
#            fig.savefig(save_fig_name, dpi=100)
#        return
#
#    def peaks(self, make_plot = False):
#        wavenumbers, counts = self.data()
#        s = savgol_filter(counts, 3, 2)
#        s -= s.min()
#        s /= s.max()
#        p = find_peaks(s,
#            height=.1,
#            threshold=None,
#            distance=2,
#            prominence=.05,
#            width=None,
#            wlen=None,
#            rel_height=0.1,
#            plateau_size=None)
#        P = pd.DataFrame(p[1])
#        P["peak pos [1/cm]"] = wavenumbers[p[0]]
#        cols = P.columns.tolist()
#        cols = [cols[-1], cols[0], cols[1], cols[2], cols[3]]
#        P = P.reindex(columns=cols)
#        P["left_bases"] = wavenumbers[P["left_bases"]]
#        P["right_bases"] = wavenumbers[P["right_bases"]]
#        #P = P.sort_values(by=["peak pos [1/cm]"])
#        self.bands = P
#        if make_plot:
#            fig, ax = plt.subplots(figsize=[8,4])
#            plt.plot(wavenumbers, counts, 'k-')
#            for pos in p[0]:
#                k_pos = wavenumbers[pos]
#                band_y = counts[pos]
#                ax.annotate(str(int(k_pos)), xy=(k_pos, band_y + counts.max()*.01),  xycoords='data',
#                            xytext=(k_pos, band_y + counts.max()*.1), textcoords='data',
#                            arrowprops=dict(facecolor='black', shrink=0.05, width=2,headwidth=5),
#                            horizontalalignment='center', verticalalignment='bottom', size=10
#                            )
#            plt.ylim(-.05, counts.max()+ counts.max()*.2)
#            plt.xlabel('Raman shift [1/cm]')
#            plt.ylabel(self.metadata["spectral unit"])
#            plt.grid(axis='x', which='both', linestyle=':')
#            plt.yticks([])
#            plt.show()
#        return
#    
#    def save(self, save_file_name):
#        with open(save_file_name + ".cha", 'wb') as f:
#            pickle.dump(self, f)
#        return
#    
#    def base(self, lam=5e5, p=0.001, niter=100, smooth=7, show=True):
#       # After Eilers, P. H., & Boelens, H. F. (2005). Baseline correction with asymmetric least squares smoothing. Leiden University Medical Centre Report, 1(1), 5.
#       # Fits & returns a background model
#       # Cut rayleigh artifact
#       _, spec = self.data()
#       #y = spec[...,50:]
#       b = spec.copy()
#       y = spec.copy()
#       if smooth > 0: y = wiener(y, smooth)
#       L = len(y)
#       D = sparse.csc_matrix(np.diff(np.eye(L), 2))
#       w = np.ones(L)
#       for i in range(niter):
#           W = sparse.spdiags(w, 0, L, L)
#           Z = W + lam * D.dot(D.transpose())
#           z = spsolve(Z, w*y)
#           w = p * (y > z) + (1-p) * (y < z)
#       #b[:50] = np.ones(50)*z[0]
#       #b[50:] = z
#       b = z
#       self.background_model = b
#       return
#
#def load(file_name):
#    with open(file_name, 'rb') as f:
#        C = pickle.load(f)
#    return C
#
## ## class chag (chada group)
#
## chag.__init__
#
## chag.compare
#
## chag.decompose
