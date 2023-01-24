# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.2 (Prototype), May 5, 2021
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