# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.2 (Prototype), May 5, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from ramanchada.analysis.peaks import fit_spectrum_peaks_pos
from ramanchada.utilities import lims


def raman_x_calibration(target_spectrum, reference_peak_list, fitmethod):
    # spectrum is a RamanSpectrum
    # reference is a list of precise reference x peak positions
    # pos must be in x range an non-nan
    reference_peak_list = [p for p in reference_peak_list
                           if p>target_spectrum.x.min() if p<target_spectrum.x.max()]
    target_pos = fit_spectrum_peaks_pos(target_spectrum.x, target_spectrum.y,
                           reference_peak_list, fitmethod)[0]
    # sort out positions where any of the fits results in nan
    ind = np.isnan(target_pos).tolist()
    target_pos = [target_pos[ii] for ii, cond in enumerate(ind) if not cond]
    reference_peak_list = [reference_peak_list[ii] for ii, cond in enumerate(ind) if not cond]
    # construct calibration
    return construct_calibration(target_pos, np.array(reference_peak_list) - np.array(target_pos))

def raman_x_calibration_from_spectrum(target_spectrum, reference_spectrum, fitmethod, peak_pos=[]):
    # Find peaks in target
    # if np peak list is given, locate peaks
    if peak_pos == []:
        target_spectrum.peaks(fit=False)
        peak_pos = target_spectrum.bands['position'].tolist()
    if fitmethod == 'hqi':
        shifts = raman_x_shifts_by_hqi(target_spectrum, peak_pos, reference_spectrum)
        return construct_calibration(peak_pos, shifts)
    # Fit corresponding peaks in reference
    reference_pos = fit_spectrum_peaks_pos(reference_spectrum.x,
                                            reference_spectrum.y, peak_pos, fitmethod)[0].tolist()
    return raman_x_calibration(target_spectrum, reference_pos, fitmethod)

def raman_y_calibration_from_spectrum(target_spectrum, reference_spectrum, x_min=-1e9, x_max=1e9):
    ref, tar = deepcopy(reference_spectrum), deepcopy(target_spectrum)
    # crop ref to intersection
    x_range_min = np.max([tar.x.min(), ref.x.min(), x_min])
    x_range_max = np.min([tar.x.max(), ref.x.max(), x_max])
    # crop reference to range
    ref.x_crop(x_range_min, x_range_max)
    ref.x_crop(x_range_min, x_range_max)
    # interpolate target to ref x
    tar.interpolate_x(ref)
    # calculate relative gain
    gain = ref.y / tar.y
    ok = ~(np.isnan(gain) ^ np.isinf(gain))
    g, x = gain[ok], tar.x[ok]
    return construct_calibration(x, g, x_col_name='Raman shift', y_col_name='y gain')

def construct_calibration(pos, shifts, x_col_name='Raman shift', y_col_name='RS correction'):
    from ramanchada.classes import RamanCalibration
    caldata = pd.DataFrame()
    caldata[x_col_name] = pos
    caldata[y_col_name] = shifts
    return RamanCalibration(caldata)

# def raman_x_calibration_from_spectrum(target_spectrum, reference_spectrum, fitmethod, peak_pos=[]):
#     # Find peaks in target
#     if fitmethod == 'hqi': peak_fit_method = 'par'
#     else: peak_fit_method = fitmethod
#     from classes import RamanCalibration
#     caldata = pd.DataFrame()
#     # if np peak list is given, locate peaks
#     if peak_pos == []:
#         target_spectrum.peaks(fitmethod=peak_fit_method)
#         target_pos = target_spectrum.bands[peak_fit_method + ' fitted position'].tolist()
#     else:
#         target_pos = fit_spectrum_peaks_pos(target_spectrum.x,
#                                     target_spectrum.y, peak_pos, peak_fit_method)[0]
#     # use unly peaks within the reference range and non-nan
#     target_pos = [p for p in target_pos if p>reference_spectrum.x.min() if p<reference_spectrum.x.max()]
#     # For calibration by HQI
#     if fitmethod == 'hqi':
#         caldata['RS correction'] = raman_x_shifts_by_hqi(target_spectrum, target_pos, reference_spectrum)
#     # For "standard" peak position calibration
#     else:
#         # Fit corresponding peaks in reference
#         reference_pos = fit_spectrum_peaks_pos(reference_spectrum.x,
#                                                 reference_spectrum.y, target_pos, fitmethod)[0]
#         # sort out positions where any of the fits results in nan
#         ind = np.isnan(target_pos) + np.isnan(reference_pos)
#         ind = ind.tolist()
#         target_pos = [target_pos[ii] for ii, cond in enumerate(ind) if not cond]
#         reference_pos = [reference_pos[ii] for ii, cond in enumerate(ind) if not cond]
#         caldata['RS correction'] = np.array(reference_pos)-np.array(target_pos)
#     caldata['Raman shift'] = target_pos
#     return RamanCalibration(caldata)

def raman_x_shifts_by_hqi(target_spectrum, anchor_points, reference_spectrum):
    align_params = [anchor_points, reference_spectrum, target_spectrum]
    bounds = [(-20, 20)] * len(anchor_points)
    ret = dual_annealing(hqi_minimize, bounds=bounds, args=align_params)
    return ret.x

def hqi_minimize(shifts_at_anchors, anchors_x_pos, REF, TAR, poly_degree=3):
    T = deepcopy(TAR)
    # calc shifts poly from anchor pos
    #shifts = np.poly1d(np.polyfit(anchors_x_pos, shifts_at_anchors, poly_degree))(T.x)
    shifts = interp1d(anchors_x_pos, shifts_at_anchors, kind="quadratic", bounds_error=False, fill_value=0)(T.x)
    # substitute target x with shifted x
    T.x += shifts
    # interpolate back on ref x & return inverse hqi
    return 1. / T.hqi(REF)

# def raman_mtf_from_psfs(psf_list):
#     mtf_sum = np.zeros(1024)
#     for peak in psf_list:
#         mtf = np.abs(np.fft.fft(peak))
#         k = np.flip( 1./( np.arange(len(mtf))+1 ) )
#         # interpolate to x (common length)
#         interp_k = interp1d(k, mtf, kind="quadratic")
#         k = np.linspace(0, 1, 1024)
#         mtf_sum += interp_k(k)
#     mtf_avg = mtf_sum/len(psf_list)
#     return mtf_avg / mtf_avg.max()

def raman_mtf_from_psfs(psf_list):
    # x must be equally spaced!
    # make all peaks same x length
    # what is the largest length?
    max_length = len(max(psf_list, key=len))
    # Nyquist frequency and lowest spatial frequency. Nyquist is 1/1 px since unit is pixels!
    k_low = 1. / max_length
    # construct resciprocal space vectors
    k = np.linspace(k_low, 1, max_length)
    k_int = np.linspace(0, 1, 1024)
    mtf_sum = np.zeros(1024)
    for psf in psf_list:
        # bring to max_length by appending last value
        psf = np.append(psf, np.ones(max_length-len(psf))*psf[-1] )
        # calc CTF
        ctf = np.abs(np.fft.fft(psf))
        # interpolate to x (common length)
        interp_k = interp1d(k, ctf, kind="quadratic", bounds_error=False, fill_value='extrapolate')
        mtf_sum += interp_k(k_int)
    mtf_avg = mtf_sum/len(psf_list)
    return k_int, mtf_avg / mtf_avg.max()

def deconvolve_mtf(y, mtf, gauss_filter_sigma=1):
    # interpolate mtf to target spectrum length
    interp_k = interp1d(np.arange(len(mtf)), mtf, kind="quadratic")
    k = np.linspace(0, len(mtf)-1, len(y))
    mtf = interp_k(k)
    # deconvolute in Fourier space
    fx = np.fft.fft(y)
    fx_dc = fx / mtf
    y_dc = np.real(np.fft.ifft(fx_dc))
    # apply Gauss filter and eliminate negatives
    y_dc = gaussian_filter1d(y_dc,  gauss_filter_sigma)
    y_dc[y_dc<0] = 0
    return y_dc

def relative_ctf(x, y_tar, y_ref):
    # Nyquist frequency and lowest spatial frequency. Nyquist is 1/1 px since unit is pixels!
    k_low = 1. / (x[-1]-x[0])
    ny = 1. / np.mean(np.diff(x))
    # construct resciprocal space vectors
    k = np.linspace(k_low, ny, len(x))
    # calc relative CTF
    rel_ctf = np.abs(np.fft.fft(y_ref) / np.fft.fft(y_tar))
    return k, rel_ctf / rel_ctf.max()

def apply_relative_ctf(x, y, rel_k, rel_ctf):
    # Nyquist frequency and lowest spatial frequency. Nyquist is 1/1 px since unit is pixels!
    k_low = 1. / (x[-1]-x[0])
    ny = 1. / np.mean(np.diff(x))
    # construct resciprocal space vectors
    k = np.linspace(k_low, ny, len(x))
    # interpolate mtf to target spectrum length
    interp_k = interp1d(rel_k, rel_ctf, kind="quadratic", bounds_error=False, fill_value='extrapolate')
    rel_ctf_int = interp_k(k)
    # deconvolute in Fourier space
    f = np.fft.fft(y)
    f_corr = f * rel_ctf_int
    y_corr = np.real(np.fft.ifft(f_corr))
    # eliminate negatives
    y_corr[y_corr<0] = 0
    return y_corr
    
def extract_xrays(SPEC):
    xray_data = []
    S = deepcopy(SPEC)
    S.fit_baseline()
    S.remove_baseline()
    S.peaks()
    # xray is amongs the 10 highest peaks
    P = S.bands.sort_values(by='intensity', ascending=False)[:10]
    # iterate through peaks
    for pos, fwhm in zip(P['position'], P['FWHM']):
        # Cut out peak
        l = lims(S.x, pos-fwhm, pos+fwhm)
        peak = l(S.y)
        # calc n of values in peak greater than half amplitude. This value is usually 1-2 for xrays.
        n_peak_points = sum(peak > peak.max()*.5)
        if n_peak_points <= 2:
            peak -= peak.min()
            peak /= peak.max()
            xray_data.append(peak)
    return xray_data


# def raman_x_shifts_by_hqi(target_spectrum, anchor_points, reference_spectrum):
#     # Maximize HQI by simulated annealing
#     align_params = [anchor_points, target_spectrum, reference_spectrum]
#     bounds = [(-20, 20)] * len(anchor_points)
#     ret = dual_annealing(align_score, bounds=bounds, args=align_params, seed=1234)
#     shifts_df = pd.DataFrame(columns=['Raman shift', 'RS correction'],
#                       data=zip(anchor_points, ret.x))
#    return shifts_df

# def align_score(shifts_at_peaks, anchor_points, target_spectrum, reference_spectrum):
#     # Make RamanCalibration from shifts_at_peaks, anchor_points
#     shifts_df = pd.DataFrame(columns=['Raman shift', 'RS correction'],
#                           data=zip(anchor_points, shifts_at_peaks))
#     from classes import RamanCalibration
#     cal = RamanCalibration(shifts_df)
#     # Calibrating target_spectrum applies shifts
#     target_spectrum_copy = deepcopy(target_spectrum)
#     target_spectrum_copy.calibrate(cal)
#     # Return reciprocal HQI with reference
#     return 1. / target_spectrum_copy.hqi(reference_spectrum)

# def raman_x_shifts_by_peaklist(spectrum, anchor_points, reference):
#     r = reference[:]
#     shifts = []
#     for ap in anchor_points:
#         if r == []: break
#         # look for index of closest point in reference list
#         closest_index = min(range(len(r)), key=lambda i: abs(r[i] - ap))
#         # append [position, shift] pair to shifts
#         shifts.append([ap, r[closest_index] - ap])
#         # delete used reference point
#         del r[closest_index]
#     shifts_df = pd.DataFrame(columns=['Raman shift', 'RS correction'], data=shifts)
#     return shifts_df