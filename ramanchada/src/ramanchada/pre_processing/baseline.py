# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.2 (Prototype), May 5, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

import numpy as np
import pandas as pd
from scipy.signal import wiener
from scipy import sparse
from scipy.sparse.linalg import spsolve
from copy import deepcopy

from ramanchada.utilities import lims


def baseline_model(y, method, **kwargs):
    methods = {'als': baseline_als, 'snip': baseline_snip}
    return methods[method](y)
   
def baseline_als(y, lam=1e5, p=0.001, niter=100, smooth=7):
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

def baseline_snip(y0, niter=30):
    # y can't have negatives. fix by offset:
    y_offset = y0.min()
    y = y0 - y_offset
    # Spectrum must be row of a DataFrame
    raman_spectra = pd.DataFrame().append(pd.DataFrame(y).T)
    spectrum_points = len(raman_spectra.columns)
    raman_spectra_transformed = np.log(np.log(np.sqrt(raman_spectra +1)+1)+1)
    working_spectra = np.zeros(raman_spectra.shape)
    for pp in np.arange(1,niter+1):
        r1 = raman_spectra_transformed.iloc[:,pp:spectrum_points-pp]
        r2 = (np.roll(raman_spectra_transformed,-pp,axis=1)[:,pp:spectrum_points-pp] + np.roll(raman_spectra_transformed,pp,axis=1)[:,pp:spectrum_points-pp])/2
        working_spectra = np.minimum(r1,r2)
        raman_spectra_transformed.iloc[:,pp:spectrum_points-pp] = working_spectra
    baseline = (np.exp(np.exp(raman_spectra_transformed)-1)-1)**2 -1
    # Re-convert to np.array and apply inverse y offset to baseline
    return baseline.to_numpy()[0].T + y_offset

def xrays(SPEC):
    S = deepcopy(SPEC)
    xray_model = np.zeros_like(S.y)
    S.fit_baseline()
    S.remove_baseline()
    S.peaks()
    # if no bands are found
    if len(S.bands) < 1:
        return np.zeros_like(S.x)
    # xray is amongs the 10 highest peaks
    P = S.bands.sort_values(by='intensity', ascending=False).head(10)
    # iterate through peaks
    for pos, fwhm in zip(P['position'], P['FWHM']):
        # Cut out peak
        l = lims(S.x, pos-fwhm, pos+fwhm)
        # calc n of values in peak greater than half amplitude. This value is usually 1-2 for xrays.
        n_peak_points = sum(l(S.y) > l(S.y).max()*.5)
        if n_peak_points <= 2:
            # get indices of peak
            _, _, pos_ind = np.intersect1d(l(S.x), S.x, return_indices=True)
            # add xray to model
            xray_model[pos_ind] = S.y[pos_ind]
    return xray_model



# def xrays(y, threshold=10):
#     # Apply string Wiener filter
#     y_w = wiener(y, 77)
#     # The difference is largest at an xray position
#     diff = y - y_w
#     pos = diff > threshold*np.std(diff)
#     # Substitute ONLY xray with smoothed values
#     y[pos] = y_w[pos]
#     return y

# def xrays(y, n_std=3, env_size=2):
#     y_d = np.abs(np.diff(y))
#     threshold = np.mean(y_d) + n_std * np.std(y_d)
#     xray = y_d > threshold
#     y_corr = y.copy()
#     for ii, channel in enumerate(xray):
#         if channel and ii-env_size>=0 and ii+env_size<len(y):
#             y_corr[ii-env_size : ii+env_size] = y_corr[ii-2*env_size : ii-env_size].mean()
#     return y_corr