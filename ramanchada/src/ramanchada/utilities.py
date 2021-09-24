# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.2 (Prototype), May 5, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

import numpy as np



def hqi(y1, y2):
    # Hit quality index (equivalent to cross-correlation)
    # See Rodriguez, J.D., et al., Standardization of Raman spectra for transfer of spectral libraries across different
    # instruments. Analyst, 2011. 136(20): p. 4232-4240.
    return np.linalg.norm(np.dot(y1, y2))**2 / np.linalg.norm(y1)**2 / np.linalg.norm(y2)**2

def lims(x, x_min, x_max):
    y_min, y_max = np.argmin(np.abs(x-x_min)), np.argmin(np.abs(x-x_max))
    def l(Y):
        return Y[...,y_min:y_max]
    return l

def interpolation_within_bounds(cal_x, cal_y, poly_degree):
    coeffs = np.polyfit(cal_x, cal_y, poly_degree)
    def interp(x):
        shifts = np.poly1d(coeffs)(x)
        over, under = x > cal_x.min()-30, x < cal_x.max()+30
        if not np.any(over*under):
            return np.zeros_like(x)
        if not np.all(over):
            shifts[~over] = shifts[over][0]
        if not np.all(under):
            shifts[~under] = shifts[under][-1]
        return shifts
    return interp