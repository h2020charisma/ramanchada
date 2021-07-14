# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Feb 26, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# Python libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import wiener
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ramanchada import chada_io 


def plotData(x_data, y_data, labels, ylabel = "Intensity", save_fig_name = "", leg=True):
    fig = plt.figure(figsize=[8,4])
    for l, y in zip(labels, y_data):
        plt.plot(x_data, y, label=l)
    plt.ylabel(ylabel)
    plt.xlabel("Raman shift [rel. 1/cm]")
    plt.grid(axis='x', which='both', linestyle=':')
    if leg: plt.legend()
    #plt.yticks([])
    if save_fig_name == "":
        plt.show()
    else:
        fig.savefig(save_fig_name, dpi=100)
    return

def lims(Y, x, x_min, x_max):
    y_min, y_max = np.argmin(np.abs(x-x_min)), np.argmin(np.abs(x-x_max))
    return Y[...,y_min:y_max]

def spec_shift(y0, x0, shifts, show=False):
    f_inter = interp1d(x0+shifts, y0, kind="cubic", bounds_error=False, fill_value=0)
    y_shifted = f_inter(x0)
    if show:
        plt.figure()
        plt.plot(x0, y0, label="original")
        plt.plot(x0, y_shifted, label="Shifted")
        plt.legend()
    return y_shifted

def stats(x_data, y_data):
    stats = {
        "Raman data type": getYDataType(y_data),
        "xy dimensions":  y_data.shape[1:],
        "no. of channels": y_data.shape[0],
        "minimum wavelength": x_data.min(),
        "maximum wavelength": x_data.max(),
        "mean counts": y_data.mean(),
        "standard deviation": y_data.std(),  
        }
    return stats

def baseline(y, lam=1e5, p=0.001, niter=100, smooth=7):
       if smooth > 0: y = wiener(y, smooth)
       L = len(y)
       D = sparse.csc_matrix(np.diff(np.eye(L), 2))
       w = np.ones(L)
       for i in range(niter):
           W = sparse.spdiags(w, 0, L, L)
           Z = W + lam * D.dot(D.transpose())
           z = spsolve(Z, w*y)
           w = p * (y > z) + (1-p) * (y < z)
       return z
   
def findRamanPeaks(x, y, sg=11, k_min=25, k_max=10000, fit=True,
              sort_by='prominence', d=200, make_plot=False, fit_plot=False):
    s = savgol_filter(y, sg, 2)
    s -= s.min()
    s /= s.max()
    y_range = y.max() - y.min()
    peaks = find_peaks(s,
        #height=.1,
        height=.01,
        threshold=None,
        distance=2,
        prominence=.05,
        width=1,
        wlen=None,
        rel_height=0.5,
        plateau_size=None)
    P = pd.DataFrame()
    P['position'] = x[peaks[0]]
    P['prominence'] = peaks[1]['prominences']
    P['FWHM'] = peaks[1]['widths']*((x.max() - x.min())/len(x))
    P['Gauss area'] = P['FWHM']/2.3548*(2*np.pi)**.5
    if fit:
        x_max = []
        for x0, w in zip(peaks[0], peaks[1]['widths']):
            x1, x2 = np.max([0, int(x0-1.*w)]), np.min([int(x0+1.*w)+1, len(x)-1])
            x_max.append(interpolatePeakFFT(x[x1:x2], y[x1:x2], d=d, show=fit_plot))
        P['fitted pos.'] = x_max
    if make_plot:
        ylabel = "Intensity"
        fig, ax = plt.subplots(figsize=[8,4])
        print(len(x), len(y))
        plt.plot(x, y, 'k-')
        x1, x2 = np.array(peaks[1]['left_ips']).astype(int), np.array(peaks[1]['right_ips']).astype(int)
        plt.hlines(y[peaks[0]]-peaks[1]['prominences']*.5*y_range, x[x1+1], x[x2+1],
                   color='k', linestyle=':')
        for p in peaks[0]:
            k_pos = x[p]
            band_y = y[p]
            ax.annotate(str(int(k_pos)), xy=(k_pos, band_y + y.max()*.01),  xycoords='data',
                        xytext=(k_pos, band_y + y.max()*.1), textcoords='data',
                        arrowprops=dict(facecolor='black', shrink=0.05, width=2,headwidth=5),
                        horizontalalignment='center', verticalalignment='bottom', size=10
                        )
        plt.ylim(-.05, y.max()+ y.max()*.2)
        plt.xlabel('Raman shift [1/cm]')
        plt.ylabel(ylabel)
        plt.grid(axis='x', which='both', linestyle=':')
        plt.show()
    return P.sort_values(by=[sort_by], ascending=False)

def fitRamanPeaks(x, s, peak_pos, peak_widths, d=200, fit_plot=False):
        x_max = []
        for x0, w in zip(peak_pos, peak_widths):
            x1, x2 = np.max([0, int(x0-1.*w)]), np.min([int(x0+1.*w)+1, len(x)-1])
            x_max.append(interpolatePeakFFT(x[x1:x2], s[x1:x2], d=d, show=fit_plot))
        return x_max
   
def interpolatePeakFFT(x, y0, pad=2000, show=False, d=100):
    # Normalize
    y = y0 - np.min(y0)
    y /= np.max(y)
    min_x, max_x = x.min(), x.max()
    # Even length is bad for FFT centering.
    if len(x)%2 == 0:
         # Truncate
         y = y[:-1]
         x = x[:-1]
    ## Tranform intensities into fourier domain
    y_f = np.fft.fft(y)
    # Pad middle (large periodicities) with zeros
    mid_pos = len(y_f)//2+2
    zeropad = np.zeros(pad)
    ext_y_f = np.hstack((y_f[:mid_pos], zeropad, y_f[mid_pos:]))
    ## Inverse FFT & normalize
    y_if = np.real(np.fft.ifft(ext_y_f))
    y_if -= np.min(y_if)
    y_if /= np.max(y_if)
    ## Create new x-axis
    #min_x, max_x = x.min(), x.max()
    ext_x = np.linspace(min_x, max_x, len(y_if))
    x1 = ext_x[np.argmax(y_if)-d:np.argmax(y_if)+d]
    y1 = y_if[np.argmax(y_if)-d:np.argmax(y_if)+d]
    z = np.polyfit(x1, y1, 2)
    p = np.poly1d(z)
    if show:
        plt.figure(figsize=[8,4])
        plt.plot(x, y, label='original data')
        plt.plot(ext_x, y_if, 'k:', label='resampled')
        plt.plot(x1, p(x1), 'r-', label='poly2 fit')
        plt.ylabel("Norm. intensity")
        plt.xlabel("Raman shift [rel. 1/cm]")
        plt.grid(axis='x', which='both', linestyle=':')
        plt.legend()
        plt.show()
    return x1[np.argmax(p(x1))]

#def plotData(x_data, y_data, labels, ylabel = "Intensity", save_fig_name = "", leg=True):
#    fig = plt.figure(figsize=[8,4])
#    for l, y in zip(labels, y_data):
#        plt.plot(x_data, y, label=l)
#    plt.ylabel(ylabel)
#    plt.xlabel("Raman shift [rel. 1/cm]")
#    plt.grid(axis='x', which='both', linestyle=':')
#    if leg: plt.legend()
#    #plt.yticks([])
#    if save_fig_name == "":
#        plt.show()
#    else:
#        fig.savefig(save_fig_name, dpi=100)
#    return
#
#def lims(Y, x, x_min, x_max):
#    y_min, y_max = np.argmin(np.abs(x-x_min)), np.argmin(np.abs(x-x_max))
#    return Y[...,y_min:y_max]
#
#def spec_shift(y0, x0, shifts, show=False):
#    f_inter = interp1d(x0+shifts, y0, kind="cubic", bounds_error=False, fill_value=0)
#    y_shifted = f_inter(x0)
#    if show:
#        plt.figure()
#        plt.plot(x0, y0, label="original")
#        plt.plot(x0, y_shifted, label="Shifted")
#        plt.legend()
#    return y_shifted
#
#def stats(x_data, y_data):
#    stats = {
#        "Raman data type": chada_io.getYDataType(y_data),
#        "xy dimensions":  y_data.shape[1:],
#        "no. of channels": y_data.shape[0],
#        "minimum wavelength": x_data.min(),
#        "maximum wavelength": x_data.max(),
#        "mean counts": y_data.mean(),
#        "standard deviation": y_data.std(),  
#        }
#    return stats
#
#def baseline(y, lam=1e5, p=0.001, niter=100, smooth=7):
#    if smooth > 0: y = wiener(y, smooth)
#    L = len(y)
#    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
#    w = np.ones(L)
#    for i in range(niter):
#        W = sparse.spdiags(w, 0, L, L)
#        Z = W + lam * D.dot(D.transpose())
#        z = spsolve(Z, w*y)
#        w = p * (y > z) + (1-p) * (y < z)
#    return z
#
#def interpolatePeakFFT(x, y0, pad=2000, show=False, d=100):
#    # Normalize
#    y = y0 - np.min(y0)
#    y /= np.max(y)
#    min_x, max_x = x.min(), x.max()
#    # Even length is bad for FFT centering.
#    if len(x)%2 == 0:
#         # Truncate
#         y = y[:-1]
#         x = x[:-1]
#    ## Tranform intensities into fourier domain
#    y_f = np.fft.fft(y)
#    # Pad middle (large periodicities) with zeros
#    mid_pos = len(y_f)//2+2
#    zeropad = np.zeros(pad)
#    ext_y_f = np.hstack((y_f[:mid_pos], zeropad, y_f[mid_pos:]))
#    ## Inverse FFT & normalize
#    y_if = np.real(np.fft.ifft(ext_y_f))
#    y_if -= np.min(y_if)
#    y_if /= np.max(y_if)
#    ## Create new x-axis
#    #min_x, max_x = x.min(), x.max()
#    ext_x = np.linspace(min_x, max_x, len(y_if))
#    x1 = ext_x[np.argmax(y_if)-d:np.argmax(y_if)+d]
#    y1 = y_if[np.argmax(y_if)-d:np.argmax(y_if)+d]
#    z = np.polyfit(x1, y1, 2)
#    p = np.poly1d(z)
#    if show:
#        plt.figure(figsize=[8,4])
#        plt.plot(x, y, label='original data')
#        plt.plot(ext_x, y_if, 'k:', label='resampled')
#        plt.plot(x1, p(x1), 'r-', label='poly2 fit')
#        plt.ylabel("Norm. intensity")
#        plt.xlabel("Raman shift [rel. 1/cm]")
#        plt.grid(axis='x', which='both', linestyle=':')
#        plt.legend()
#        plt.show()
#    return x1[np.argmax(p(x1))]