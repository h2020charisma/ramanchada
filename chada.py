# Python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tempfile import TemporaryFile
from scipy.signal import wiener, savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
import os
import pickle
# Third party spectrum readers 
from renishawWiRE import WDFReader
from specio import specread


class Chada():
    def __init__(self, file):
        try:
            f = open(file, "rb", buffering=0)
            self.binary_data = f.read()
        except Exception as err:
            raise err 
        finally:    
            f.close()
        self.readers = {'.spc': specread, '.wdf': WDFReader}
        self.filename = file[file.rfind(os.path.sep)+2:]
        ext = file[file.rfind("."):]
        reader = self.readers[ext]
        s = reader(file)
#        # This is for .wdf (Renishaw) files only
        counts = s.spectra
        wavenumbers = s.xdata
#        counts = s.amplitudes
#        wavenumbers = s.wavelength
        if np.mean(np.diff(wavenumbers)) < 0:
            counts = np.flip(counts)
            wavenumbers = np.flip(wavenumbers)
        self.metadata = {
        "original filepath": file,
        "native format": ext,
        "laser wavelength": s.laser_length,
        "no. of accumulations": s.accumulation_count,
        "spectral unit": s.spectral_unit.name,
        "OEM software name": s.application_name,
        "OEM software version": s.application_version,
        "minimum wavelength": wavenumbers.min(),
        "maximum wavelength": wavenumbers.max(),
        "no. of channels": len(counts),
        "mean counts": counts.mean(),
        "standard deviation": counts.std(),  
        }
        return
    
    def meta(self):
        return pd.DataFrame.from_dict(self.metadata, orient='index', columns=['value'])
    
    def data(self):
        try:
            specfile = TemporaryFile("wb", delete=False)
            specfile.write(self.binary_data)
            ext = self.metadata["native format"]
        except Exception as err:
            raise err
        finally:    
            specfile.close()
        try:    
            reader = self.readers[ext]
            s = reader(specfile.name)
        except Exception as err:
            raise err
        finally:
            s.close()
        os.unlink(specfile.name)
#        counts = s.amplitudes
#        wavenumbers = s.wavelength
#        if np.mean(np.diff(wavenumbers)) < 0:
#            counts = np.flip(counts)
#            wavenumbers = np.flip(wavenumbers)
        counts = s.spectra
        wavenumbers = s.xdata
        if np.mean(np.diff(wavenumbers)) < 0:
            counts = np.flip(counts)
            wavenumbers = np.flip(wavenumbers)
        if hasattr(self, 'background_model'):
            counts -= self.background_model
        return wavenumbers, counts

    def plot(self, save_fig_name = ""):
        wavenumbers, counts = self.data()
        ylabel = self.metadata["spectral unit"]
        if hasattr(self, 'background_model'):
            ylabel = ylabel + ", background-corrected"
        fig = plt.figure(figsize=[8,4])
        plt.plot(wavenumbers, counts, 'k-')
        plt.ylabel(ylabel)
        plt.xlabel("Raman shift [rel. 1/cm]")
        plt.grid(axis='x', which='both', linestyle=':')
        plt.yticks([])
        if save_fig_name == "":
            plt.show()
        else:
            fig.savefig(save_fig_name, dpi=100)
        return

    def peaks(self, make_plot = False):
        wavenumbers, counts = self.data()
        s = savgol_filter(counts, 3, 2)
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
        P["peak pos [1/cm]"] = wavenumbers[p[0]]
        cols = P.columns.tolist()
        cols = [cols[-1], cols[0], cols[1], cols[2], cols[3]]
        P = P.reindex(columns=cols)
        P["left_bases"] = wavenumbers[P["left_bases"]]
        P["right_bases"] = wavenumbers[P["right_bases"]]
        #P = P.sort_values(by=["peak pos [1/cm]"])
        self.bands = P
        if make_plot:
            fig, ax = plt.subplots(figsize=[8,4])
            plt.plot(wavenumbers, counts, 'k-')
            for pos in p[0]:
                k_pos = wavenumbers[pos]
                band_y = counts[pos]
                ax.annotate(str(int(k_pos)), xy=(k_pos, band_y + counts.max()*.01),  xycoords='data',
                            xytext=(k_pos, band_y + counts.max()*.1), textcoords='data',
                            arrowprops=dict(facecolor='black', shrink=0.05, width=2,headwidth=5),
                            horizontalalignment='center', verticalalignment='bottom', size=10
                            )
            plt.ylim(-.05, counts.max()+ counts.max()*.2)
            plt.xlabel('Raman shift [1/cm]')
            plt.ylabel(self.metadata["spectral unit"])
            plt.grid(axis='x', which='both', linestyle=':')
            plt.yticks([])
            plt.show()
        return
    
    def save(self, save_file_name):
        with open(save_file_name + ".cha", 'wb') as f:
            pickle.dump(self, f)
        return
    
    def base(self, lam=5e5, p=0.001, niter=100, smooth=7, show=True):
       # After Eilers, P. H., & Boelens, H. F. (2005). Baseline correction with asymmetric least squares smoothing. Leiden University Medical Centre Report, 1(1), 5.
       # Fits & returns a background model
       # Cut rayleigh artifact
       _, spec = self.data()
       #y = spec[...,50:]
       b = spec.copy()
       y = spec.copy()
       if smooth > 0: y = wiener(y, smooth)
       L = len(y)
       D = sparse.csc_matrix(np.diff(np.eye(L), 2))
       w = np.ones(L)
       for i in range(niter):
           W = sparse.spdiags(w, 0, L, L)
           Z = W + lam * D.dot(D.transpose())
           z = spsolve(Z, w*y)
           w = p * (y > z) + (1-p) * (y < z)
       #b[:50] = np.ones(50)*z[0]
       #b[50:] = z
       b = z
       self.background_model = b
       return

def load(file_name):
    with open(file_name, 'rb') as f:
        C = pickle.load(f)
    return C

# ## class chag (chada group)

# chag.__init__

# chag.compare

# chag.decompose
