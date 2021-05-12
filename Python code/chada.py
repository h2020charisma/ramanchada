# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.2 (Prototype), May 5, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# =============This file contains the Chada classes============================

# Python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import wiener, savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
import os
import sys
import zipfile
import h5py
import time
import ast
from scipy.interpolate import interp1d
from sklearn.decomposition import NMF

sys.path.append(r'/utilities')
from chada_utilities import lims, plotData, stats, spec_shift, baseline

# =========================CLASSES===========================================
class Chada():
    def __init__(self, chada_path, verbose=True):
        self.path = chada_path
        try:
            # Open HDF5
            f = h5py.File(chada_path, "r")
            # Load data
            a = f["Raman data"].attrs
            self.metadata = dict(zip(a.keys(), [str(v) for v in a.values()]))
            self.x_data_0, self.y_data_0 = f["Raman data"][0], f["Raman data"][1]
            self.x_data, self.y_data = self.x_data_0.copy(), self.y_data_0.copy()
            transformers = [ast.literal_eval(i.decode('utf-8')) for i in f["Transformers"]]
            self.commits = [i.decode('utf-8') for i in f["Commits"]]
        except Exception as err:
            raise err
        finally:    
            f.close()
        # self.transformers will be populated upon transformers execution
        self.transformers = []
        for t in transformers:
            # t[0] has transformer function name
            transformer_func = getattr(self, t[0])
            # if t has parameters, pass them to func
            if len(t) > 1: transformer_func(*t[1:])
            else: transformer_func()
        return
    
    def commit(self, commit = "Updated CHADA on " + time.ctime() ):
        self.commits.append(commit)
        f = h5py.File(self.path, "r+")
        # Update dynamic metadata
        f["Raman data"].attrs.update(self.metadata)
        #for key in self.metadata:
        #    f["Raman data"].attrs[key] = self.metadata[key]
        # replace transformers
        del f["Transformers"], f["Commits"]
        f["Transformers"] = [str(tr) for tr in self.transformers]
        f["Commits"] = [str(co) for co in self.commits]
        f.close()
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
        #self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
        return
    
    #--------------------------UTILITIES------------------------
    def plot(self, save_fig_name = "", original=False):
        if original: x, y = self.x_data_0, self.y_data_0
        else: x, y = self.x_data, self.y_data
        ylabel = "Intensity"
        fig = plt.figure(figsize=[8,4])
        plt.plot(x, y, 'k-')
        plt.ylabel(ylabel)
        plt.xlabel("Raman shift [rel. 1/cm]")
        plt.grid(axis='x', which='both', linestyle=':')
        #plt.yticks([])
        if save_fig_name == "":
            plt.show()
        else:
            fig.savefig(save_fig_name, dpi=100)
        return
    
    def statistics(self, original=False):
        s = pd.Series( stats (self.x_data, self.y_data) )
        print(s)
        if original:
            s = pd.Series( stats (self.x_data_0, self.y_data_0) )
            print(s)
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
    def fit_baseline(self, lam=1e5, p=0.001, niter=100, smooth=7, show=False):
       # After Eilers, P. H., & Boelens, H. F. (2005). Baseline correction with asymmetric least squares smoothing. Leiden University Medical Centre Report, 1(1), 5.
       # Fits & returns a background model
       self.baseline = baseline(self.y_data,lam=lam,p=p,niter=niter,smooth=smooth)
#       y = self.y_data.copy()
#       if smooth > 0: y = wiener(y, smooth)
#       L = len(y)
#       D = sparse.csc_matrix(np.diff(np.eye(L), 2))
#       w = np.ones(L)
#       for i in range(niter):
#           W = sparse.spdiags(w, 0, L, L)
#           Z = W + lam * D.dot(D.transpose())
#           z = spsolve(Z, w*y)
#           w = p * (y > z) + (1-p) * (y < z)
#       self.baseline = z
       if show:
           ylabel = "Intensity"
           plt.figure(figsize=[8,4])
           plt.plot(self.x_data, self.y_data, 'k')
           plt.plot(self.x_data, self.baseline, 'r')
           plt.ylabel(ylabel)
           plt.xlabel("Raman shift [rel. 1/cm]")
           plt.grid(axis='x', which='both', linestyle=':')
       return
   
    def remove_baseline(self, b=[]):
        if b != []: self.baseline = np.array(b)
        self.y_data -= self.baseline
        self.transformers.append([ 'remove_baseline', self.baseline.tolist() ])
        return

    def x_crop(self, k_min=300, k_max=1800):
        self.transformers.append(['x_crop', k_min, k_max])
        self.y_data = lims(self.y_data, self.x_data, k_min, k_max)
        self.x_data = lims(self.x_data, self.x_data, k_min, k_max)
        #self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
        return
    
    def normalize(self, normalization_type = 'area'):
        self.transformers.append(['normalize', normalization_type])
        norms = {'area': [np.min,np.mean], 'minmax': [np.min,np.max], 'vector':[np.mean,np.std]}
        func1, func2 = norms[normalization_type]
        self.y_data -= func1(self.y_data, axis=0)
        self.y_data /= func2(self.y_data, axis=0)
        return
    
    def XCalFromFile(self, calibration_file, show=False):
        try:
            # Open archive
            zf = zipfile.ZipFile(calibration_file)
            # Load data
            x_pos = ast.literal_eval(zf.read("peak_pos.txt").decode('utf-8'))
            shifts_pos = ast.literal_eval(zf.read("shifts_at_peaks.txt").decode('utf-8'))
        except Exception as err:
            raise err
        finally:    
            zf.close()
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
        #self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
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
        #self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
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
        #self.dynamic_metadata = dynamicMetaDataUpdate(self.x_data, self.y_data)
        return


class ChadaGroup():
    def __init__(self, chada_files, transformers=[]):
        Y = []
        X = []
        x_increment_all = np.inf
        x_min_all = -np.inf
        x_max_all = np.inf
        self.labels = []
        for file in chada_files:
            G = Chada(file)
            if transformers != []:
                # reset Chada to original state
                G.rewind(0)
                # excute group transformers
                for t in transformers:
                    if t != []:
                        # t[0] has transformer function name
                        transformer_func = getattr(G, t[0])
                        # if t has parameters, pass them to func
                        if len(t) > 1: transformer_func(*t[1:])
                        else: transformer_func()
            X.append(G.x_data)
            Y.append(G.y_data)
            #x_increment = np.abs(np.diff(G.x_data)).min()
            x_min, x_max = G.x_data.min(), G.x_data.max()
            x_min_all = np.max([x_min_all, x_min])
            x_max_all = np.min([x_max_all, x_max])
            #x_increment_all = np.min([x_increment_all, x_increment])
            del G
            self.labels.append(os.path.basename(file))
        x_increment_all = 1
        self.x_data = np.arange(x_min_all, x_max_all, x_increment_all)
        interpolated_Y = []
        for x, y in zip(X,Y):
            f_inter = interp1d(x, y, kind="cubic")
            interpolated_Y.append(f_inter(self.x_data))
        self.y_data = np.array(interpolated_Y)
        del X, Y, interpolated_Y
        return
    
    def plot(self, save_fig_name = "", leg=True):
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
    
    def base(self, lam=1e5, p=0.001, niter=100, smooth=7, show=False):
        for ii, y in enumerate(self.y_data):
            self.y_data[ii,:] -= baseline(y, lam=lam, p=p, niter=niter, smooth=smooth)
        if show:
            self.plot()
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