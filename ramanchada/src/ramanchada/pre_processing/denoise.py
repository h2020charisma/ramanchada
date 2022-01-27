# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.2 (Prototype), May 5, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

import statsmodels.api as sm
import numpy as np

from scipy.signal import wiener, savgol_filter, medfilt
from scipy.signal.windows import boxcar
from scipy.ndimage import gaussian_filter1d


def smooth_curve(y, method, *args, **kwargs):
    methods = {
        'sg': savgol_filter,
        'wiener': wiener,
        'median': medfilt,
        'gauss': gaussian_filter1d,
        'lowess': smooth_lowess,
        'boxcar': smooth_boxcar
        }
    return methods[method](y, *args, **kwargs)


def smooth_lowess(y, span=11):
    x = np.linspace(0, 1, len(y))
    return sm.nonparametric.lowess(y, x, frac=(5*span / len(y)), return_sorted=False)

def smooth_boxcar(y, box_pts=11):
    box = boxcar(box_pts, sym=True)
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth