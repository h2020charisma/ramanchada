# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Feb 26, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# Python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter, peak_widths, find_peaks_cwt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

from ramanchada.utilities import lims


def find_spectrum_peaks(x, y, prominence=.05, sg=11, sort_by='prominence'):
    """
    find_spectrum_peaks(x, y, prominence=.05, sg=11, x_min=25, x_max=10000, sort_by='prominence')
    Finds peaks and their FWHM and return in spectral units
    """
    # Crop to specified range
    #l = lims(x, x_min, x_max)
    #x, y = l(x), l(y)
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
    P['Gauss area'] = P['intensity']*P['FWHM']/2.3548*(2*np.pi)**.5
    return P.sort_values(by=[sort_by], ascending=False, ignore_index=True)

def find_spectrum_peaks_cwt(x, y, width=10, sort_by='intensity'):
    """
    find_spectrum_peaks_cwt(x, y, width=10, sort_by='prominence')
    Finds peak positions using wavelet transformation.
    """
    #y -= y.min()
    #y /= y.max()
    integer_positions = find_peaks_cwt(y, width, wavelet=None, max_distances=None, gap_thresh=None, min_length=None, min_snr=1, noise_perc=10, window_size=None)
    P = pd.DataFrame()
    P['position'] = x[integer_positions]
    P['intensity'] = y[integer_positions]
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
    methods = {'voigt': voigt_fit, 'lorentz': lorentz_fit, 'pearson': pearson4_fit, 
        'beta': beta_profile_fit, 'par': parabola_fit, 'gl': gauss_lorentz_fit, 'dvoigt': double_voigt_fit}
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
    y0 = y.min()
    amp = (y.max()-y.min()) * 10
    cen = x[y.argmax()]
    # sigma should be ~1/2 of FWHM
    fwhm = find_spectrum_peaks(x, y)['FWHM'][0]
    sigma = .5*fwhm
    gamma = .5*fwhm
    # Do fit
    try:
        popt, pcov = curve_fit(voigt, x, y, p0=[y0, amp, cen, sigma, gamma])
        pars = popt
        fit_x = np.linspace(x.min(), x.max(), 1000)
        error = np.sum(np.abs(voigt(x, *pars) - y))/np.sum(y)
        fitted_y = voigt(fit_x, *pars)
        fitpos = fit_x[fitted_y.argmax()]
        # Approx. for width (Olivero and Longbothum)
        fwhm_lorentz = 2*pars[4]
        fwhm_gauss = np.sqrt(8*np.log(2)) * pars[3]
        width = 2* 0.5346*fwhm_lorentz + np.sqrt(0.2166*fwhm_lorentz**2 + fwhm_gauss)
        area = pars[1]
        if show:
            plot_fit_curve(x, y, fit_x, voigt, pars, fitpos, error)
    except RuntimeError:
        return np.nan, np.nan, np.nan
    #return fitpos, width, area
    return get_function_properties(x, voigt, pars)

def lorentz_fit(x, y, show=False):
    """
    fitpos, width = voigt_fit(x, y, show=False, d=0)
    Fits a Voigt profile.
    If show is True, the fits and resulting maxima are plotted.
    The width and area are calculated analytically after Olivero and Longbothum
    """
    # Estimate starting parameters
    y0 = y.min()
    cen = x[y.argmax()]
    # sigma should be ~1/2 of FWHM
    wid = find_spectrum_peaks(x, y)['FWHM'][0]/2
    amp = (y.max()-y.min())*wid
    # Do fit
    try:
        popt, pcov = curve_fit(lorentz, x, y, p0=[y0, amp, cen, wid])
        pars = popt
        fit_x = np.linspace(x.min(), x.max(), 1000)
        error = np.sum(np.abs(lorentz(x, *pars) - y))/np.sum(y)
        fitted_y = lorentz(fit_x, *pars)
        fitpos = fit_x[fitted_y.argmax()]
        width = (pars[2]**2 + pars[3]*pars[2])**.5 - (pars[2]**2 - pars[3]*pars[2])**.5
        area = pars[1]
        if show:
            plot_fit_curve(x, y, fit_x, lorentz, pars, fitpos, error)
    except RuntimeError:
        return np.nan, np.nan, np.nan
    #return fitpos, width, area
    return get_function_properties(x, lorentz, pars)

def pearson4_fit(x, y, show=False):
    """

    """
    # Estimate starting parameters
    y0 = y.min()
    k = y.max()-y.min()
    # peak position (a1);
    a1 = x[y.argmax()]
    # width (a2);
    a2 = find_spectrum_peaks(x, y)['FWHM'][0]
    """parameter a3 determines whether the peak shape is flatter (a3>1) or
    sharper (a3<1) than a Lorentzian distribution"""
    a3 = 1
    """peak asymmetry is determined by a4, with fronted peaks having
    a positive a4 and the tailed peaks having a negative a4."""
    a4 = 0
    # Do fit
    try:
        popt, pcov = curve_fit(pearson4, x, y, p0=[y0, a1, a2, a3, a4, k])
        pars = popt
        fit_x = np.linspace(x.min(), x.max(), 1000)
        error = np.sum(np.abs(pearson4(x, *pars) - y))/np.sum(y)
        fitted_y = pearson4(fit_x, *pars)
        fitpos = fit_x[fitted_y.argmax()]
        width = pars[2]
        area = pars[5]
        if show:
            plot_fit_curve(x, y, fit_x, pearson4, pars, fitpos, error)
    except RuntimeError:
        return np.nan, np.nan, np.nan
    #return fitpos, width, area
    return get_function_properties(x, pearson4, pars)

def beta_profile_fit(x, y, show=False):
    """

    """
    # Estimate starting parameters
    y0 = y.min()
    amp = y.max()-y.min()
    # peak position (a1);
    a1 = x[y.argmax()]
    # data scaling (a2)
    a2 = find_spectrum_peaks(x, y)['FWHM'][0]/2
    """parameterized by two positive shape parameters alpha and beta:"""
    alpha = 1.5
    beta = 1.5
    # Do fit
    try:
        popt, pcov = curve_fit(beta_profile, x, y, p0=[y0, amp, alpha, beta, a1, a2])
        pars = popt
        fit_x = np.linspace(x.min(), x.max(), 1000)
        error = np.sum(np.abs(beta_profile(x, *pars) - y))/np.sum(y)
        fitted_y = beta_profile(fit_x, *pars)
        fitpos = fit_x[np.nan_to_num(fitted_y).argmax()]
        width = pars[5] # data scaling (a2)
        area = pars[1] # is equal to amp since distribution is normalized
        if show:
            plot_fit_curve(x, y, fit_x, beta_profile, pars, fitpos, np.nan_to_num(error))
    except RuntimeError:
        return np.nan, np.nan, np.nan
    #return fitpos, width, area
    return get_function_properties(x, beta_profile, pars)

def double_voigt_fit(x, y, show=False):
    """
    fitpos, width = voigt_fit(x, y, show=False, d=0)
    Fits a Voigt profile.
    If show is True, the fits and resulting maxima are plotted.
    The width and area are calculated analytically after Olivero and Longbothum
    """
    # Estimate starting parameters
    y0 = y.min()/2
    amp = (y.max()-y.min()) * 10
    cen1, cen2 = x[2], x[-3]
    # sigma should be ~1/2 of FWHM
    fwhm = find_spectrum_peaks(x, y)['FWHM'][0]/2
    sigma = .5*fwhm
    gamma = .5*fwhm
    # Do fit
    try:
        popt, pcov = curve_fit(double_voigt, x, y, p0=[y0, amp, cen1, sigma, gamma, amp, cen2, sigma, gamma])
        pars = popt
        fit_x = np.linspace(x.min(), x.max(), 1000)
        error = np.sum(np.abs(double_voigt(x, *pars) - y))/np.sum(y)
        fitted_y = double_voigt(fit_x, *pars)
        fitpos = fit_x[fitted_y.argmax()]
        # Approx. for with (Olivero and Longbothum)
        fwhm_lorentz = max( [2*pars[4], 2*pars[4+4]] )
        fwhm_gauss = max( [np.sqrt(8*np.log(2)) * pars[3], np.sqrt(8*np.log(2)) * pars[3+4]] )
        width = 2* 0.5346*fwhm_lorentz + np.sqrt(0.2166*fwhm_lorentz**2 + fwhm_gauss)
        area = np.array([pars[1], pars[1+4]])
        if show:
            plot_fit_curve(x, y, fit_x, double_voigt, pars, fitpos, error)
    except RuntimeError:
        return np.nan, np.nan, np.nan
    #return fitpos, width, area
    return get_function_properties(x, double_voigt, pars)

def gauss_lorentz_fit(x, y, show=False):
    """
    fitpos, width, area = gauss_voigt_fit(x, y, show=False, d=0)
    Fits the sum of a Gauss + Lorentz, with independent parameters for each function.
    This works well for asymmetric peaks, but does not give a single analytic maximum  position.
    The maximum is returned as the absolute max of the numerical values of the fitted function.
    The width and area are geometrically estimated from the fitted function using peak_widths.
    If show is True, the fits and resulting maxima are plotted.
    """
    # Estimate starting parameters
    y0 = y.min()/2
    ampG1 = y.max()-y.min()
    cenG1 = x[y.argmax()]
    # sigma should be ~1/2 of FWHM
    fwhm = find_spectrum_peaks(x, y)['FWHM'][0]
    sigmaG1 = .5*fwhm
    ampL1 = y.max()-y.min()
    cenL1 = x[y.argmax()]
    widL1 = fwhm
    # Do fit
    try:
        popt, pcov = curve_fit(gauss_lorentz, x, y, p0=[y0, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1])
        pars = popt
        fit_x = np.linspace(x.min(), x.max(), 1000)
        error = np.sum(np.abs(gauss_lorentz(x, *pars) - y))/np.sum(y)
        fitted_y = gauss_lorentz(fit_x, *pars)
        fitpos = fit_x[fitted_y.argmax()]
        width = spectrum_peak_widths(fit_x, fitted_y, [fitpos], rel_height=.5)[0]
        area = np.trapz(fitted_y, fit_x)
        if show:
            plot_fit_curve(x, y, fit_x, gauss_lorentz, pars, fitpos, error)
    except RuntimeError:
        return np.nan, np.nan, np.nan
    #return fitpos, width, area
    return get_function_properties(x, gauss_lorentz, pars)

def parabola_fit(x, y, show=False):
    """
    fitpos, width, area =  
    Interpolates the counts to pad (~1000) x values, then fits a parabola to the central 10%.
    The interpolate is equivalent to padding in Fourier Space, and probably superflous.
    The width and area are geometrically estimated from the interpolated function using peak_widths.
    """
    new_x = np.linspace(x.min(), x.max(), 1000)
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

def get_function_properties(x, function, pars):
    # make x vector with 5x peak range
    x_range = np.ptp(x)
    x_full = np.arange(x.mean()-5*x_range, x.mean()+5*x_range)
    x_fine = np.linspace(x.min(), x.max(), 1000)
    # pars[0] is baseline. Set to zero!
    pars[0] = 0
    # calculate theoretical peak
    y_full = np.nan_to_num(function(x_full, *pars))
    y_fine = np.nan_to_num(function(x_fine, *pars))
    # area is numerical integral
    #area = np.trapz(y_fine, x_fine)
    # peak pos is at max
    peak_pos = x_fine[y_fine.argmax()]
    # get width from y_full, which is already scaled at 1 1/cm
    width, _, _, _ = peak_widths(y_full, [y_full.argmax()])
    width = width[0]
    # approximate area as Gaussian.
    area = y_fine.max() * width/2.3548*(2*np.pi)**.5
    return peak_pos, width, area

def plot_fit_curve(x, y, fit_x, function, pars, fitpos, error):
    plt.figure()
    plt.plot(x, y, 'ko', label = "data")
    plt.plot(fit_x, function(fit_x, *pars), 'r-', label = f'{function.__name__}, err = {error*100:.1f} %')
    plt.plot(fitpos, function(fitpos, *pars), 'r^')
    plt.legend()
    plt.xlabel("Raman shift [rel. 1/cm]")
    plt.ylabel("intensity")
    plt.title("Peak at " + '{:4.2f}'.format(fitpos) + "rel 1/cm")
    plt.show()

def voigt(x, y0, amp, cen, sigma, gamma):
    return y0 + amp * voigt_profile(x-cen, sigma, gamma)

def gauss(x, y0, amp, cen, sigma):
    return y0 + (amp*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp(-((x-cen)**2)/((2*sigma)**2))))

def lorentz(x, y0, amp, cen, wid):
    #return y0 + amp * 0.25/cen**2 * 1./( (x-cen)**2 + wid**2 /4 )
    #return y0 + amp * 1./( (x**2-cen**2)**2 + wid**2 * cen**2 )
    return y0 + ((amp*wid**2/((x-cen)**2+wid**2)) )

def pearson4(x, y0, a1, a2, a3, a4, k):
    xx = (x-a1)/a2
    return y0 + k*( 1+xx**2 )**-a3 * np.exp( -a4 * np.arctan(xx) )

def beta_profile(x, y0, amp, alpha, beta, a1, a2):
    X = (x-a1)/a2
    mode = (alpha-1)/(alpha+beta-2)
    term_1 = (X+mode)**(alpha-1) * (1-X-mode)**(beta-1)
    term_2 = mode**(alpha-1) * (1-mode)**(beta-1)
    return y0 + amp * term_1 / term_2

def double_voigt(x, y0, amp1, cen1, sigma1, gamma1, amp2, cen2, sigma2, gamma2):
    return voigt(x, y0, amp1, cen1, sigma1, gamma1) + voigt(x, y0, amp2, cen2, sigma2, gamma2)

def gauss_lorentz(x, y0, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
    return y0 + gauss(x, y0, ampG1, cenG1, sigmaG1) + lorentz(x, y0, ampL1, cenL1, widL1)
              
