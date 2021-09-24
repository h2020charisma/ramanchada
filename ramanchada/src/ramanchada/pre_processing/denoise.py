# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.2 (Prototype), May 5, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

from scipy.signal import wiener, savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d


def smooth_curve(y, method, **kwargs):
    methods = {'sg': savgol_filter, 'wiener': wiener, 'median': medfilt, 'gauss': gaussian_filter1d}
    return methods[method](y, **kwargs)