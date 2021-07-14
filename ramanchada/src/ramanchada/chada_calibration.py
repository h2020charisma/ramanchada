# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Feb 26, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# Python libraries
import numpy as np
import sys
from scipy.optimize import dual_annealing
from scipy.interpolate import interp1d
import os
import time
import zipfile

from ramanchada import chada,chada_utilities


def makeXCalFromSpec(target_file, reference_file, method='peaks', bounds=[-10.,10.], cal_range=[]):
    # Creates a calibration using two CHADA files: target and refereence.
    # If later the calibration is applied to the target, is will be aligned to the reference.
    T = Chada(target_file)
    R = Chada(reference_file)
    if cal_range != []:
        R.x_crop(cal_range[0], cal_range[1])
        T.x_crop(cal_range[0], cal_range[1])
    # Normalize spectra
    T.normalize()
    R.normalize()
    # Determine valid range of calibration (intersection of x axes)
    cal_upper = np.min([T.x_data.max(), R.x_data.max()])
    cal_lower = np.max([T.x_data.min(), R.x_data.min()])
    cal_range = [cal_lower, cal_upper]
    # Get peak positions from target_spectrum
    shifts_func = {'peaks': getShiftsFromPeaks, 'hqi': getShiftsFromHQI}
    peak_pos, shifts_at_peaks = shifts_func[method](R, T, bounds)
    # Save calibration as .chacal archive
    createCalFile(target_file, reference_file, peak_pos, shifts_at_peaks, cal_range)
    return

def getShiftsFromPeaks(R, T, bounds):
    # Get peak positions from reference and target_spectra
    R.peaks(fit=True, sort_by='prominence', make_plot=False, d=200,
              fit_plot = False)
    T.peaks(fit=True, sort_by='prominence', make_plot=False, d=200,
              fit_plot = False)
    peak_pos_ref = np.array(R.bands['fitted pos.'])
    peak_pos_target = np.array(T.bands['fitted pos.'])
    num_peaks = np.min([len(peak_pos_ref), len(peak_pos_target)])
    print(num_peaks)
    # Truncate peak lists to common length
    peak_pos_ref, peak_pos_target = peak_pos_ref[:num_peaks], peak_pos_target[:num_peaks]
    return peak_pos_target, peak_pos_ref - peak_pos_target

def getShiftsFromHQI(R, T, bounds):
    # Get peak positions from target_spectrum
    T.peaks(fit=True, sort_by='position', make_plot=False, d=200,
              fit_plot = False)
    peak_pos = np.array(T.bands['fitted pos.'])
    # interpolate reference to target x_data
    f_inter = interp1d(R.x_data, R.y_data, kind="cubic", bounds_error=False, fill_value=0)
    reference = f_inter(T.x_data)
    # Maximize HQI by simulated annealing
    align_params = [T.y_data, reference, T.x_data, peak_pos]
    lw = [bounds[0]] * len(peak_pos)
    up = [bounds[1]] * len(peak_pos)
    ret = dual_annealing( align_score, bounds=list(zip(lw, up)), args=align_params, seed=1234 )
    return peak_pos, ret.x

def createCalFileZip(target, target_path, reference, reference_path, bounds, peak_pos,
                  shifts_at_peaks, cal_range, interpolation_kind='cubic'):
    # Create.chacal archive
    metadata = {}
    metadata["Generated on"] = time.ctime()
    metadata["Target file"] = target_path
    metadata["Reference file"] = reference_path
    metadata["Interpolation kind"] = interpolation_kind
    metadata["Calibration x range"] = cal_range
    metadata["No of anchor points"] = len(peak_pos)
    # Write attributes to .chacal text file archive
    filename, _ = os.path.splitext(target_path)
    zf = zipfile.ZipFile(filename + ".chacal", mode="w", compression=zipfile.ZIP_DEFLATED)
    zf.writestr("peak_pos.txt", str( peak_pos.tolist() ))
    zf.writestr("shifts_at_peaks.txt", str( shifts_at_peaks.tolist() ))
    zf.writestr("metadata.txt", str(metadata))
    zf.close()
    print("Saved calibration file '" + filename + ".chacal'")
    return

def createCalFile(target_path, reference_path, peak_pos,
                  shifts_at_peaks, cal_range, interpolation_kind='cubic',
                  calibration_filename=''):
    metadata = {}
    metadata["Generated on"] = time.ctime()
    metadata["Target file"] = target_path
    metadata["Reference file"] = reference_path
    metadata["Interpolation kind"] = interpolation_kind
    metadata["Calibration x range"] = cal_range
    metadata["No of anchor points"] = len(peak_pos)
    if calibration_filename == '':
        calibration_filename, _ = os.path.splitext(target_path)
    f = h5py.File(calibration_filename + ".chacal", "w")
    # Store Raman dataset + label
    xy = f.create_dataset("Shifts", data=np.vstack((peak_pos, shifts_at_peaks)))
    xy.dims[0].label = 'Raman shift [1/cm]'
    xy.dims[1].label = 'Peak shift [1/cm]'
    # Store metadata
    xy.attrs.update(metadata)
    f.close()
    return

def align_score(shifts_at_peaks, y, y_ref, x, peak_pos):
    # extrapolate shift vector
    f_inter = interp1d(peak_pos, shifts_at_peaks, kind="cubic", bounds_error=False, fill_value="extrapolate")
    shifts = f_inter(x)
    # calculate HQI of shifted spectrum and ref
    return 1. / hqi(y_ref, spec_shift(y, x, shifts))

def hqi(y1, y2):
    # Hit quality index (equivalent to cross-correlation)
    # See Rodriguez, J.D., et al., Standardization of Raman spectra for transfer of spectral libraries across different
    # instruments. Analyst, 2011. 136(20): p. 4232-4240.
    return np.linalg.norm(np.dot(y1, y2))**2 / np.linalg.norm(y1)**2 / np.linalg.norm(y2)**2

#def makeXCalFromSpec(target_file, reference_file, bounds=[-10.,10.], cal_range=[]):
#    # Creates a calibration using two CHADA files: target and refereence.
#    # If later the calibration is applied to the target, is will be aligned to the reference.
#    T = chada.Chada(target_file)
#    R = chada.Chada(reference_file)
#    if cal_range != []:
#        R.x_crop(cal_range[0], cal_range[1])
#        T.x_crop(cal_range[0], cal_range[1])
#    # Normalize spectra
#    T.normalize()
#    R.normalize()
#    # Determine valid range of calibration (intersection of x axes)
#    cal_upper = np.min([T.x_data.max(), R.x_data.max()])
#    cal_lower = np.max([T.x_data.min(), R.x_data.min()])
#    cal_range = [cal_lower, cal_upper]
#    # Get peak positions from target_spectrum
#    T.peaks()
#    peak_pos = np.array(T.bands['peak pos [1/cm]'])
#    # interpolate reference to target x_data
#    f_inter = interp1d(R.x_data, R.y_data, kind="cubic", bounds_error=False, fill_value=0)
#    reference = f_inter(T.x_data)
#    # Maximize HQI by simulated annealing
#    align_params = [T.y_data, reference, T.x_data, peak_pos]
#    lw = [bounds[0]] * len(peak_pos)
#    up = [bounds[1]] * len(peak_pos)
#    ret = dual_annealing( align_score, bounds=list(zip(lw, up)), args=align_params, seed=1234 )
#    # Save calibration as .chacal archive
#    file_chacal = createCalFile(T, target_file, R, reference_file, bounds, peak_pos, ret.x, cal_range)
#    return peak_pos, ret.x, file_chacal
#
#def createCalFile(target, target_path, reference, reference_path, bounds, peak_pos,
#                  shifts_at_peaks, cal_range, interpolation_kind='cubic'):
#    # Create.chacal archive
#    metadata = {}
#    metadata["Generated on"] = time.ctime()
#    metadata["Target file"] = target_path
#    metadata["Reference file"] = reference_path
#    metadata["Interpolation kind"] = interpolation_kind
#    metadata["Calibration x range"] = cal_range
#    metadata["No of anchor points"] = len(peak_pos)
#    # Write attributes to .chacal text file archive
#    filename, _ = os.path.splitext(target_path)
#    zf = zipfile.ZipFile(filename + ".chacal", mode="w", compression=zipfile.ZIP_DEFLATED)
#    zf.writestr("peak_pos.txt", str( peak_pos.tolist() ))
#    zf.writestr("shifts_at_peaks.txt", str( shifts_at_peaks.tolist() ))
#    zf.writestr("metadata.txt", str(metadata))
#    zf.close()
#    print("Saved calibration file '" + filename + ".chacal'")
#    return filename + ".chacal"
#
#def align_score(shifts_at_peaks, y, y_ref, x, peak_pos):
#    # extrapolate shift vector
#    f_inter = interp1d(peak_pos, shifts_at_peaks, kind="cubic", bounds_error=False, fill_value="extrapolate")
#    shifts = f_inter(x)
#    # calculate HQI of shifted spectrum and ref
#    return 1. / hqi(y_ref, chada_utilities.spec_shift(y, x, shifts))
#
#def hqi(y1, y2):
#    # Hit quality index (equivalent to cross-correlation)
#    # See Rodriguez, J.D., et al., Standardization of Raman spectra for transfer of spectral libraries across different
#    # instruments. Analyst, 2011. 136(20): p. 4232-4240.
#    return np.linalg.norm(np.dot(y1, y2))**2 / np.linalg.norm(y1)**2 / np.linalg.norm(y2)**2
