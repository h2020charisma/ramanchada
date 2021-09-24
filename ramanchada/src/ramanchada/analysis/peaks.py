# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Feb 26, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# Python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy.optimize import curve_fit
from scipy.special import voigt_profile


def find_spectrum_peaks(x, y, prominence=.05, sg=11, x_min=25, x_max=10000, sort_by='prominence'):
    """
    find_spectrum_peaks(x, y, prominence=.05, sg=11, x_min=25, x_max=10000, sort_by='prominence')
    Finds peaks and their FWHM and return in spectral units
    """
    # Crop to specified range
    from utilities import lims
    l = lims(x, x_min, x_max)
    x, y = l(x), l(y)
    # Filter + minmax normalization are important!
    s = savgol_filter(y, sg, 2)
    s -= s.min()
    s /= s.max()
    integer_positions, props_dict = find_peaks(s,
        #height=.1,
        height=.01,
        threshold=None,
        distance=2,
        prominence=prominence,
        width=1,
        wlen=None,
        # rel_height=0.5 means FWHM
        rel_height=0.5,
        plateau_size=None)
    local_units_per_channel = x[integer_positions] - x[integer_positions-1]
    widths_in_x_units = props_dict['widths']*local_units_per_channel
    P = pd.DataFrame()
    P['position'] = x[integer_positions]
    P['intensity'] = y[integer_positions]
    P['prominence'] = props_dict['prominences']
    P['FWHM'] = widths_in_x_units
    P['Gauss area'] = P['FWHM']/2.3548*(2*np.pi)**.5
    return P.sort_values(by=[sort_by], ascending=False, ignore_index=True)

def spectrum_peak_widths(x, y, pos_in_x_units, rel_height=0.5):
    """
    spectrum_peak_widths(x, y, pos_in_x_units, rel_height=0.5)
    Gets the geometric widths of peaks at given positions.
    """
    # Translate pos_in_x_units to coordinates
    p = np.array([np.argmin(np.abs(xp-x)) for xp in pos_in_x_units])
    # Filter + minmax normalization are important!
    s = savgol_filter(y, 11, 2)
    s -= s.min()
    s /= s.max()
    w, _, _, _ = peak_widths(s, p, rel_height=rel_height)
    # Translate back to x_units
    units_per_channel = np.mean(np.diff(x))
    return w*units_per_channel
    
def fit_spectrum_peaks_pos(x, y, pos_in_x_units, method, interval_width=2, show=False):
    """
    x_fit_pos, x_fit_width = fit_spectrum_peaks_pos(x, y, pos_in_x_units, method, show=False)
    Does peak fittig using the given method.
    Returns the fitted positions and FWHMs as arrays
    """
    # pos must be in x range
    pos_in_x_units = [p for p in pos_in_x_units if p>x.min() if p<x.max()]
    widths_in_x_units = spectrum_peak_widths(x, y, pos_in_x_units, rel_height=.5)
    # 2x 1/e**2 is 1.699 x FWHM
    widths_in_x_units = np.array([max(w,10) for w in widths_in_x_units])      
    half_peak_widths = 1.699/2*widths_in_x_units
    x_fit_pos = []
    x_fit_width = []
    x_fit_area = []
    from utilities import lims
    methods = {'par': parabola_fit, 'voigt': voigt_fit, 'vg': gauss_voigt_fit}
    for p, w in zip(pos_in_x_units, half_peak_widths):
        l = lims(x, p-interval_width*w, p+interval_width*w)
        pos, width, area = methods[method](l(x), l(y), show=show)
        x_fit_pos.append(pos)
        x_fit_width.append(width)
        x_fit_area.append(area)
    return np.array(x_fit_pos), np.array(x_fit_width), np.array(x_fit_area)

def voigt_fit(x, y, show=False):
    """
    fitpos, width = voigt_fit(x, y, show=False, d=0)
    Fits a Voigt profile.
    If show is True, the fits and resulting maxima are plotted.
    The width and area are calculated analytically after Olivero and Longbothum
    """
    # Estimate starting parameters
    amp = y.max()-y.min()*10
    cen = x[y.argmax()]
    # sigma should be ~1/2 of FWHM
    fwhm = (x.max() - x.min()) / 2
    sigma = .5*fwhm
    gamma = .5*fwhm
    # Do fit
    try:
        popt, pcov = curve_fit(voigt, x, y, p0=[amp, cen, sigma, gamma])
        pars = popt
        fit_x = np.linspace(x.min(), x.max(), 1000)
        error = np.sum(np.abs(voigt(x, *pars) - y))/np.sum(y)
        fitted_y = voigt(fit_x, *pars)
        fitpos = fit_x[fitted_y.argmax()]
        # Approx. for with (Olivero and Longbothum)
        fwhm_lorentz = 2*pars[3]
        fwhm_gauss = np.sqrt(8*np.log(2)) * pars[2]
        width = 2* 0.5346*fwhm_lorentz + np.sqrt(0.2166*fwhm_lorentz**2 + fwhm_gauss)
        area = pars[0]
        if show:
            plt.figure()
            plt.plot(x, y, 'ko', label = "data")
            plt.plot(fit_x, voigt(fit_x, *pars), 'r-', label = "Voigt, err = " + str(int(error*100)) + "%")
            plt.plot(fitpos, voigt(fitpos, *pars), 'r^')
            plt.legend()
            plt.xlabel("Raman shift [rel. 1/cm]")
            plt.ylabel("intensity")
            plt.title("Peak at " + '{:4.2f}'.format(fitpos) + "rel 1/cm")
            plt.show()
    except RuntimeError:
        return np.nan, np.nan, np.nan
    return fitpos, width, area

def double_voigt_fit(x, y, show=False):
    """
    fitpos, width = voigt_fit(x, y, show=False, d=0)
    Fits a Voigt profile.
    If show is True, the fits and resulting maxima are plotted.
    The width and area are calculated analytically after Olivero and Longbothum
    """
    # Estimate starting parameters
    amp = y.max()-y.min()*10
    cen1, cen2 = x[0], x[-1]
    # sigma should be ~1/2 of FWHM
    fwhm = (x.max() - x.min()) / 2
    sigma = .5*fwhm
    gamma = .5*fwhm
    # Do fit
    try:
        popt, pcov = curve_fit(double_voigt, x, y, p0=[amp, cen1, sigma, gamma, amp, cen2, sigma, gamma])
        pars = popt
        fit_x = np.linspace(x.min(), x.max(), 1000)
        error = np.sum(np.abs(double_voigt(x, *pars) - y))/np.sum(y)
        fitted_y = double_voigt(fit_x, *pars)
        fitpos = fit_x[fitted_y.argmax()]
        # Approx. for with (Olivero and Longbothum)
        fwhm_lorentz = np.array([2*pars[3], 2*pars[3+4]])
        fwhm_gauss = np.array([np.sqrt(8*np.log(2)) * pars[2], np.sqrt(8*np.log(2)) * pars[2+4]])
        width = 2* 0.5346*fwhm_lorentz + np.sqrt(0.2166*fwhm_lorentz**2 + fwhm_gauss)
        area = np.array([pars[0], pars[0+4]])
        if show:
            plt.figure()
            plt.plot(x, y, 'ko', label = "data")
            plt.plot(fit_x, double_voigt(fit_x, *pars), 'r-', label = "Double Voigt, err = " + str(int(error*100)) + "%")
            plt.plot(fitpos, double_voigt(fitpos, *pars), 'r^')
            plt.legend()
            plt.xlabel("Raman shift [rel. 1/cm]")
            plt.ylabel("intensity")
            plt.title("Peak at " + '{:4.2f}'.format(fitpos) + "rel 1/cm")
            plt.show()
    except RuntimeError:
        return np.nan, np.nan, np.nan
    return list(fitpos), list(width), list(area)

def gauss_voigt_fit(x, y, show=False):
    """
    fitpos, width, area = gauss_voigt_fit(x, y, show=False, d=0)
    Fits the sum of a Gauss + Lorentz, with independent parameters for each function.
    This works well for asymmetric peaks, but does not give a single analytic maximum  position.
    The maximum is returned as the absolute max of the numerical values of the fitted function.
    The width and area are geometrically estimated from the fitted function using peak_widths.
    If show is True, the fits and resulting maxima are plotted.
    """
    # Estimate starting parameters
    ampG1 = y.max()-y.min()
    cenG1 = x[y.argmax()]
    # sigma should be ~1/2 of FWHM
    fwhm = (x.max() - x.min()) / 2
    sigmaG1 = .5*fwhm
    ampL1 = y.max()-y.min()
    cenL1 = x[y.argmax()]
    widL1 = .5*fwhm
    # Do fit
    try:
        popt, pcov = curve_fit(gauss_voigt, x, y, p0=[ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1])
        pars = popt
        fit_x = np.linspace(x.min(), x.max(), 1000)
        error = np.sum(np.abs(gauss_voigt(x, *pars) - y))/np.sum(y)
        fitted_y = gauss_voigt(fit_x, *pars)
        fitpos = fit_x[fitted_y.argmax()]
        width = spectrum_peak_widths(fit_x, fitted_y, [fitpos], rel_height=.5)[0]
        area = np.trapz(fitted_y, fit_x)
        if show:
            plt.figure()
            plt.plot(x, y, 'ko', label = "data")
            plt.plot(fit_x, gauss_voigt(fit_x, *pars), 'r-', label = "Voigt, err = " + str(int(error*100)) + "%")
            plt.plot(fitpos, gauss_voigt(fitpos, *pars), 'r^')
            plt.legend()
            plt.xlabel("Raman shift [rel. 1/cm]")
            plt.ylabel("intensity")
            plt.title("Peak at " + '{:4.2f}'.format(fitpos) + "rel 1/cm")
            plt.show()
    except RuntimeError:
        return np.nan, np.nan, np.nan
    return fitpos, width, area

def parabola_fit(x, y, show=False):
    """
    fitpos, width, area =  
    Interpolates the counts to pad (~1000) x values, then fits a parabola to the central 10%.
    The interpolate is equivalent to padding in Fourier Space, and probably superflous.
    The width and area are geometrically estimated from the interpolated function using peak_widths.
    """
    new_x = np.linspace(x.min(), x.max(), 1000)
    from scipy.interpolate import interp1d
    f = interp1d(x, y, kind="quadratic", bounds_error=False, fill_value=0)
    new_y = f(new_x)
    #w, _, _, _ = peak_widths(new_y, np.array(new_y.argmax()), rel_height=0.5)
    w = len(new_x)
    left_lim, right_lim = max(0, np.argmax(new_y)-w//10), min(len(new_x), np.argmax(new_y)+w//10)
    x1 = new_x[left_lim:right_lim]
    y1 = new_y[left_lim:right_lim]
    z = np.polyfit(x1, y1, 2)
    p = np.poly1d(z)
    fitpos = new_x[np.argmax(p(new_x))]
    if show:
        plt.figure(figsize=[8,4])
        plt.plot(x, y, 'ko', label='original data')
        plt.plot(new_x, new_y, 'k:', label='resampled')
        plt.plot(x1, p(x1), 'r-', label='poly2 fit')
        plt.ylabel("intensity")
        plt.xlabel("Raman shift [rel. 1/cm]")
        plt.grid(axis='x', which='both', linestyle=':')
        plt.legend()
        plt.show()
    width = spectrum_peak_widths(new_x, new_y, [fitpos], rel_height=.5)[0]
    area = np.trapz(new_y, new_x)
    return fitpos, width, area

def voigt(x, amp, cen, sigma, gamma):
    return amp * voigt_profile(x-cen, sigma, gamma)

def double_voigt(x, amp1, cen1, sigma1, gamma1, amp2, cen2, sigma2, gamma2):
    return voigt(x, amp1, cen1, sigma1, gamma1) + voigt(x, amp2, cen2, sigma2, gamma2)

def gauss_voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/((2*sigmaG1)**2)))) +\
              ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)) )
              
