# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.2 (Prototype), May 5, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# external imports
from weakref import ref
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import deepcopy
import os
import time
import matplotlib.style as plot_style
import seaborn as sns
# ramanchada imports
from ramanchada.decorators import specstyle, log, change_y, change_x, mark_peaks
from ramanchada.pre_processing.baseline import baseline_model, xrays
from ramanchada.pre_processing.denoise import smooth_curve
from ramanchada.file_io.io import import_native,\
    read_chada, create_chada_from_native, commit_chada, write_new_chada
from ramanchada.analysis.peaks import find_spectrum_peaks, fit_spectrum_peaks_pos, find_spectrum_peaks_cwt
from ramanchada.analysis.signal import snr
from ramanchada.utilities import hqi, lims, interpolation_within_bounds, labels_from_filenames, wavelengths_to_wavenumbers
from ramanchada.calibration.calibration import raman_x_calibration, raman_x_calibration_from_spectrum, raman_y_calibration_from_spectrum,\
    deconvolve_mtf, relative_ctf, apply_relative_ctf, raman_mtf_from_psfs, extract_xrays, construct_calibration

    
class Curve:
    """
    Basic Data class with designated x and y columns. Can be spectra, distributions, chromatograms, ...
    """
    def __init__(self, data, x_column_name, y_column_name):
        """
        Parameters
        ----------
        data : pandas DataFrame
            >Data is imported as a DataFrame. This can come from common data formats such as csv or Excel.
            
        x_column_name : str
            >Name of the column holding the x data.
            
        y_column_name : str
            >Name of the column holding the y data.  

        Returns
        -------
        None.

        """
        # data is a pd.DataFrame
        self.x = np.array(data[x_column_name])
        self.y = np.array(data[y_column_name])
        self.x_label, self.y_label = x_column_name, y_column_name
        self.time = time.ctime()
    def __repr__(self):
        info = f'{self.__class__.__name__} with {len(self.x)} points generated {self.time}' + '\n' +\
                f'{self.x_label}: {self.x.min()} - {self.x.max()}'+ '\n' +\
                f'{self.y_label}: {self.y.min()} - {self.y.max()}'
        return info
    @specstyle
    def plot(self):
        """
        Plot the curve, i.e. the y data against x data.

        Returns
        -------
        None.

        """
        plt.plot(self.x, self.y)
    @change_y
    @log
    def normalize(self, method='snv', x_min=-1e9, x_max=1e9):
        """
        Normalize y data by subtracting a constant, followed by dividing by another constant.
        Both constants are calculated from the y data.

        Parameters
        ----------
        method : str, optional
            > Normalization method as described in *Ryabchykov, O., Guo, S., & Bocklitz, T. (2019). Analyzing Raman spectroscopic data. Physical Sciences Reviews, 4(2), 1–16. https://doi.org/10.1515/psr-2017-0043.*  
            - 'snv': standard normal variate - subtract mean, divide by standard deviation  
            - 'vector': vector normalization  - subtract 0, divide by vector norm   
            - 'minmax': min-max scaling - subtract minimun, divide by maximum  
            - 'area': integrated intensity scaling - subtract minimun, divide by mean  
            The default is 'snv'.  
            
        x_min : double, optional
            > Lower x boundary of interval where constants are determined from. The default is -1e9.
            
        x_max : double, optional. 
            > Upper x boundary of interval where constants are determined from. The default is 1e9.

        Returns
        -------
        None.

        """
        methods = {'snv': [np.mean, np.std],
                   'vector': [np.zeros_like, np.linalg.norm],
                   'minmax': [np.min, np.max],
                   'area': [np.min, np.mean]}
        # normalization is measured within specified interval
        l = lims(self.x, x_min, x_max)
        self.y -= methods[method][0](l(self.y))
        self.y /= methods[method][1](l(self.y))
    @log
    def smooth(self, method='sg', *args, **kwargs):
        """
        Smoothing / denoising of x data.

        Parameters
        ----------
        method : str, optional
            > Smoothing method. Can be one of the following:

        **- 'sg': Savitzky-Golay filter** - see scipy.signal.savgol_filter
        
        **kwargs

        window_length : int, optional
        > Window Length for Savitzky-Golay filter. The default is 11.
        
        polyorder : int, optional
        > Polynomial order for Savitzky-Golay filter. The default is 3.

        **- 'wiener': Wien filter** - see scipy.signal.wiener
        
        **kwargs

        mysize : int or array_like, optional
        > A scalar or an N-length list giving the size of the Wiener filter window in each dimension. Elements of mysize should be odd. If mysize is a scalar, then this scalar is used as the size in each dimension.

        noise : float, optional
        > The noise-power to use. If None, then noise is estimated as the average of the local variance of the input.

        **- 'median': median filter** - see scipy.signal.medfilt
        
        **kwargs

        kernel_size : array_like, optional
        > A scalar or an N-length list giving the size of the median filter window in each dimension. Elements of kernel_size should be odd. If kernel_size is a scalar, then this scalar is used as the size in each dimension. Default size is 3 for each dimension.

        **- 'gauss': Gauss filter** - see scipy.ndimage.gaussian_filter1d

        sigma : scalar
        > Standard deviation for Gaussian kernel

        **- 'lowess': Locally Weighted Scatterplot Smoothing (LOWESS)** - see statsmodels.nonparametric.smoothers_lowess.lowess
        
        **kwargs

        span : float
        > Width of interval (in channels) to use for estimating each y-value.

        **- 'boxcar': filter by rectangular window or Dirichlet window**
        
        **kwargs

        box_pts : int
        > Number of points in the output window. If zero or less, an empty array is returned.

        The default is 'sg'.
            
        Returns
        -------
        None.

        """
        if len(args) == 0 and len(kwargs) == 0:
            kwargs = {'window_length': 11, 'polyorder': 3}
        self.y = smooth_curve(self.y, method, *args, **kwargs)
    @log
    def x_crop(self, x_min, x_max):
        """
        Crop data on x axis.

        Parameters
        ----------
        x_min : double
            > Lower x boundary of interval in x units
            
        x_max : double
            > Upper x boundary of interval in x units.

        Returns
        -------
        None.

        """
        l = lims(self.x, x_min, x_max)
        self.x, self.y = l(self.x), l(self.y)
    def to_csv(self, filepath):
        DF = pd.DataFrame( {self.x_label: self.x, self.y_label: self.y} )
        DF.to_csv(filepath)
    def to_excel(self, filepath):
        DF = pd.DataFrame( {self.x_label: self.x, self.y_label: self.y} )
        DF.to_excel(filepath)
    
class Spectrum(Curve):
    """
    A curve with peaks that can be located and analyzed.
    Inherits from Curve.
    """
    def __init__(self, data, x_column_name, y_column_name, label=None,
                 x_label='spectral index', y_label='intensity [a.u.]'):
        """

        Parameters
        ----------
        data : pandas DataFrame
            > Data is imported as a DataFrame. This can come from common data formats such as csv or Excel.
            
        x_column_name : str
            > Name of the column holding the x data.
            
        y_column_name : str
            > Name of the column holding the y data.
            
        x_label : str, optional
            > Label for x data. The default is 'spectral index'.
            
        y_label : TYPE, optional
            > Label for y data The default is 'intensity [a.u.]'.

        Returns
        -------
        None.

        """
        super().__init__(data, x_column_name, y_column_name)
        self.x_label, self.y_label = x_label, y_label
        
    @change_y
    @log
    def invert(self):
        """
        Flips y axis by inversion followed by minimum subtraction.

        Returns
        -------
        None.

        """
        self.y *= -1.
        self.y -= self.y.min()

    def peaks(self, prominence=0.05, x_min=-1e9, x_max=1e9, cwt=False, fit=True, fitmethod = 'voigt', interval_width=2, cwt_width=20,
              sort_by='prominence', show=False):
        """
        Automated detection and analysis of peaks.

        Parameters
        ----------
        x_min : double, optional
            > Lower x boundary of search interval in x units. The default is -1e9.
            
        x_max : double, optional
            > Upper x boundary of search interval in x units. The default is 1e9.
            
        fit : bool, optional
            > True if peak parameters should be determined by fitting an analytical model. The default is True.
            
        fitmethod : str, optional
            > Model to fit to each peak.   
            - 'par': parabola  
            - 'voigt': Voigt distribution  
            - 'vg': Sum of independent Gauss + Voigt model  
            The default is 'voigt'.
            
        interval_width : double, optional
            > The interval width, in FWHMs, on which each peak is fitted. The default is 2.
            
        sort_by : str, optional
            > The column after which peaks are sorted in the .bands attribute. The default is 'prominence'.
            
        show : bool, optional
            > True if fits should be plotted with the data for each peak. The default is False.

        Returns
        -------
        None.

        """
        l = lims(self.x, x_min, x_max)
        x, y = l(self.x), l(self.y)
        if cwt:
            self.bands = find_spectrum_peaks_cwt(x, y, width=cwt_width, sort_by='intensity')
        else:
            self.bands = find_spectrum_peaks(x, y, prominence=prominence, sort_by=sort_by)
        if fit:
            positions, widths, areas, positions_error, widths_error, areas_error = \
                fit_spectrum_peaks_pos(x, y, self.bands['position'], method = fitmethod,\
                    interval_width=interval_width, show=show)
            self.bands[fitmethod + ' fitted position'] = positions
            self.bands[fitmethod + ' fitted FWHM'] = widths
            self.bands[fitmethod + ' fitted area'] = areas
            self.bands[fitmethod + ' position error'] = positions_error
            self.bands[fitmethod + ' FWHM error'] = widths_error
            self.bands[fitmethod + ' area error'] = areas_error
    @specstyle
    @mark_peaks
    def show_bands(self):
        if not hasattr(self, 'bands'):
            print(f'No bands located yet. Use *.peaks()* first.')
            return
        else:
            plt.plot(self.x, self.y)
            self.bands.style
    def fit_baseline(self, method='als', lam=1e5, p=0.001, niter=100, smooth=7):
        """
        Fits a flourescent background model.

        Parameters
        ----------
        method : str, optional
            > Model fitting method as described in  
            *Ryabchykov, O., Guo, S., & Bocklitz, T. (2019). Analyzing Raman spectroscopic data. Physical Sciences Reviews, 4(2), 1–16. https://doi.org/10.1515/psr-2017-0043.*  
            - 'als': Asymmetric least squares.  
            After *He, S., Zhang, W., Liu, L., Huang, Y., He, J., Xie, W., … P. W.-A., & 2014,  undefined. (n.d.). Baseline correction for Raman spectra using an improved asymmetric least squares method. Pubs.Rsc.Org.*  
            - 'snip': statistics-sensitive non-linear iterative peak clipping.  
            After *Caccia, M., Ebolese, A., Maspero, M., Santoro, R., Tecnologia, A., Locatelli, M., Pieracci, M., Tintori, C., & Caen, S. A. (n.d.). Background removal procedure based on the SNIP algorithm for γ − ray spectroscopy with the CAEN Educational Kit. CAEN Tools for Discovery Educational Note ED3163, 1–4.*    
            The default is 'als'.
            
        lam : double, optional
            > lambda parameter for ALS. The default is 1e5.
            
        p : double, optional
            > p parameter for ALS. The default is 0.001.
            
        niter : int, optional
            > Number of iterations. The default is 100.
            
        smooth : int, optional
            > Kernel length for Wien filtering prior to ALS baseline fitting. The default is 7.

        Returns
        -------
        None.

        """
        self.baseline = baseline_model(self.y, method=method, lam=lam, p=p, niter=niter, smooth=smooth)
    @specstyle
    def plot_baseline(self):
        plt.plot(self.x, self.baseline, 'r-')
        plt.plot(self.x, self.y, 'k:')
    @log
    def remove_baseline(self, method='als', lam=1e5, p=0.001, niter=100, smooth=7):
        """
        Fits and removes a flourescent background model.

        Parameters
        ----------
        method : str, optional
            > Model fitting method as described in  
            *Ryabchykov, O., Guo, S., & Bocklitz, T. (2019). Analyzing Raman spectroscopic data. Physical Sciences Reviews, 4(2), 1–16. https://doi.org/10.1515/psr-2017-0043.*  
            - 'als': Asymmetric least squares.  
            After *He, S., Zhang, W., Liu, L., Huang, Y., He, J., Xie, W., … P. W.-A., & 2014,  undefined. (n.d.). Baseline correction for Raman spectra using an improved asymmetric least squares method. Pubs.Rsc.Org.*  
            - 'snip': statistics-sensitive non-linear iterative peak clipping.  
            After *Caccia, M., Ebolese, A., Maspero, M., Santoro, R., Tecnologia, A., Locatelli, M., Pieracci, M., Tintori, C., & Caen, S. A. (n.d.). Background removal procedure based on the SNIP algorithm for γ − ray spectroscopy with the CAEN Educational Kit. CAEN Tools for Discovery Educational Note ED3163, 1–4.*    
            The default is 'als'.
            
        lam : double, optional
            > lambda parameter for ALS. The default is 1e5.
            
        p : double, optional
            > p parameter for ALS. The default is 0.001.
            
        niter : int, optional
            > Number of iterations. The default is 100.
            
        smooth : int, optional
            > Kernel length for Wien filtering prior to ALS baseline fitting. The default is 7.

        Returns
        -------
        None.

        """
        self.fit_baseline(method='als', lam=1e5, p=0.001, niter=100, smooth=7)
        self.y -= self.baseline
            
    @change_x
    @log
    def interpolate_x(self, reference_spectrum=[]):
        """
        Interpolate spectrum onto the x axis of a reference spectrum. If no reference is given, the x axis is interpolated to an increment of 1.0 units (usually rel 1/cm).

        Parameters
        ----------
        reference_spectrum : Spectrum, optional
            > Reference spectrum. The default is None.

        Returns
        -------
        None.

        """
        # If no reference is given, just sample to one wavenumber
        if reference_spectrum == []:
            x = np.arange(np.floor( self.x.min() ), np.ceil( self.x.max() )+1, 1)
        else:
            x = reference_spectrum.x.copy()
        # reference_spectrum is a Spectrum
        f_inter = interp1d(self.x, self.y, kind="quadratic", bounds_error=False, fill_value=0)
        self.y = f_inter(x)
        self.x = x.copy()
    def hqi(self, reference_spectrum):
        """
        Calculates the Hit Quality Index after *Rodriguez, J. D., Westenberger, B. J., Buhse, L. F., & Kauffman, J. F. (2011b). Standardization of Raman spectra for transfer of spectral libraries across different instruments. Analyst, 136(20), 4232–4240. https://doi.org/10.1039/c1an15636e*  

        Parameters
        ----------
        > reference_spectrum : Spectrum
            Reference Spectrum with which HQI is calculated.

        Returns
        -------
        double
            > Hit Quality Index

        """
        # Make a copy to avoid changing data upon interpolate_x
        self_copy = deepcopy(self)
        self_copy.interpolate_x(reference_spectrum)
        return hqi(self_copy.y, reference_spectrum.y)
    def math(self, spectrum, operator='+'):
        ops = {'+': np.add,
            '-': np.subtract,
            '*': np.multiply,
            '/': np.divide}
        s = deepcopy(spectrum)
        s.interpolate_x(self)
        self.y = ops[operator](self.y, s.y)


class RamanSpectrum(Spectrum):
    """
    Class for a Raman Spectrum, which can have metadata and can be calibrated.
    Inherits from *Spectrum*.
    """
    def __init__(self, data, x_column_name, y_column_name,
                 x_label='Raman Shift [rel 1/cm]', y_label='intensity [a.u.]'):
        """

        Parameters
        ----------
        data : pandas DataFrame
            > Data is imported as a DataFrame. This can come from common data formats such as csv or Excel.
            
        x_column_name : str
            > Name of the column holding the x data.
            
        y_column_name : str
            > Name of the column holding the y data.
            
            
            #      x_label : str, optional
            > Label for x data. The default is 'spectral index'.
            
        y_label : TYPE, optional
            > Label for y data The default is 'intensity [a.u.]'.

        Returns
        -------
        None.

        """
        super().__init__(data, x_column_name, y_column_name)
        self.x_label, self.y_label = x_label, y_label
        self.time = time.ctime()
        self.meta = {}
    @log
    def add_metadata(self, meta_dict):
        """
        Adds metadata to a RamanSpectrum.

        Parameters
        ----------
        meta_dict : dict
            > Python dictionary, containing pairs of key: value.

        Returns
        -------
        None.

        """
        self.meta.update(meta_dict)
    def reset_x(self):
        self.x = np.arange( len(self.y) )
        self.x_label = 'Spectrum channel no.'
    def make_x_axis(self, x_peak_positions_dict, x_unit='Raman shift [rel. 1/cm]', show=False, column='position',order=3):
        # Note: peaks must be discovered first with .peaks() method, so that .bands attribute exists.
        # merge x_peak_positions_dict with 'position' column of .bands by index
        found_positions = self.bands[column]
        selected_found_positions = np.array( [found_positions[peak_index] for peak_index in x_peak_positions_dict.keys()] )
        reference_positions = np.array([v for v in x_peak_positions_dict.values()])
        # generate & return RamanCalibration with polynomial degree=1 (linear interpolation)
        axis_data = pd.DataFrame( {'original x: ' + self.x_label: selected_found_positions, x_unit: reference_positions} )
        x_axis = RamanCalibration(data=axis_data, poly_degree=order, interpolate=True)
        if show:
            x_axis.show()
        return x_axis
    @change_x
    @log
    def assign_x(self, x_axis):
        self.x = x_axis.interp_x(self.x)
        self.x += x_axis.x_offset_value
        self.x_label = x_axis.y_label
    @change_x
    @log
    def calibrate(self, calibration):
        """
        Calibrate the x axis.

        Parameters
        ----------
        calibration : RamanCalibration
            > Calibration object for the x  axis.

        Returns
        -------
        None.

        """
        # Interpolate Raman shift corrections
        x_shifts = calibration.interp_x(self.x)
        # Substitute with new x
        self.x += x_shifts
        # Make sure x values remain sorted
        inds = self.x.argsort()
        self.x, self.y = self.x[inds], self.y[inds]
    def make_x_calibration(self, reference, fitmethod = 'voigt', peak_pos=[], poly_degree=3, interpolate=False):
        """
        Generate an x axis calibration, either to a RamanSpectrum or a list of exact reference peak positions.

        Parameters
        ----------
        reference : RamanSpectrum or list
            > Reference for calibration.
            If RamanSpectrum, the reference must be a spectrum recorded using the same sample, and calibrated.
            If list, the reference must be a list of exact peak positions.
            
        fitmethod : str, optional
            > Model to fit to each peak of Target and Reference  
            - 'par': parabola  
            - 'voigt': Voigt distribution  
            - 'vg': Sum of independent Gauss + Voigt model  
            The default is 'voigt'.

        peak_pos : list, optional
            > Only if reference is a RamanSpectrum: peak positions that should be fit and included in the calibration.
            The default is [].

        Returns
        -------
        RamanCalibration
            > Calibration object containing the relative shifts as well as a polynomial interpolation.

        """
        # Returns a RamanCalibration
        # If a Raman spectrum is given as refernece
        if reference.__class__.__name__ in ['RamanSpectrum', 'RamanChada']:
            return raman_x_calibration_from_spectrum(self, reference, fitmethod=fitmethod, peak_pos=peak_pos,
                poly_degree=poly_degree, interpolate=interpolate)
        # If a list of peak positions is given
        if reference.__class__.__name__ == 'list':
            print('list')
            return raman_x_calibration(self, reference, fitmethod=fitmethod, poly_degree=poly_degree, interpolate=interpolate)
        else:
            return None
    @change_y
    @log
    def calibrate_y(self, calibration):
        """
        Calibrate the y axis.

        Parameters
        ----------
        calibration : RamanCalibration
            > Calibration object for the y  axis.

        Returns
        -------
        None.

        """
        # Interpolate Raman gain corrections
        gain = calibration.interp_x(self.x)
        if np.all(gain == 0):
            print('Calibration not in data range!')
            return
        # Apply gain correction
        self.y *= gain
    def make_y_calibration(self, reference, x_min=-1e9, x_max=1e9):
        """
        Generate an y axis calibration to a RamanSpectrum.

        Parameters
        ----------
        reference : RamanSpectrum
            > The reference must be a spectrum recorded using the same sample, and calibrated in x and y.
            
        x_min : double
            > Lower x boundary of calibration interval in x units
            
        x_max : double
            > Upper x boundary of calibration interval in x units.

        Returns
        -------
        RamanCalibration
            > Calibration object containing the relative gain as well as a polynomial interpolation.

        """
        return raman_y_calibration_from_spectrum(self, reference, x_min=x_min, x_max=x_max)
    @change_y
    @log
    def deconvolve_MTF(self, mtf, gauss_filter_sigma=1):
        """
        Deconvolves an MTF from a RamanSpectrum.

        Parameters
        ----------
        mtf : RamanMTF
            > MTF object containing the MTF model in Fourier space.
            
        gauss_filter_sigma : double, optional
            > Sigma of a Gauss filter applied after deconvolution to reduce excessive noise. The default is 1.

        Returns
        -------
        None.

        """
        self.y = deconvolve_mtf(self.y, mtf.y, gauss_filter_sigma=gauss_filter_sigma)
    def make_res_calibration(self, reference):
        """
        Calibrate the resolution (peak broadening) to that of a reference.

        Parameters
        ----------
        reference : RamanSpectrum
            > The reference must be a spectrum recorded using the same sample, and calibrated in x and y.

        Returns
        -------
        RamanCTF
            > Object containing the model of ther relative point spread function in Fourier space.

        """
        # use copies to not mess with data here
        ref = deepcopy(reference)
        tar = deepcopy(self)
        # crop ref to intersection
        x_min = np.max([tar.x.min(), ref.x.min()])
        x_max = np.min([tar.x.max(), ref.x.max()])
        ref.x_crop(x_min, x_max)
        # interpolate both to common x with delta = 1/cm
        ref.interpolate_x()
        tar.interpolate_x(ref)
        # normalize
        ref.normalize('minmax')
        tar.normalize('minmax')
        # calc rel. CTF
        rel_k, rel_ctf = relative_ctf(ref.x, tar.y, ref.y)
        ctf_data = pd.DataFrame()
        ctf_data['spatial frequency'] = rel_k
        ctf_data['amplitude'] = rel_ctf
        return RamanCTF(ctf_data)
    @change_x
    @change_y
    @log
    def set_resolution(self, rel_ctf):
        """
        Apply a relative CTF to a RamanSpectrum

        Parameters
        ----------
        rel_ctf : RamanCTF
            > Object containing the model of ther relative point spread function in Fourier space.
            When applied, the resolution is approximated to that of the reference instrument with which the CTF was calibrated.

        Returns
        -------
        None.

        """
        # x values must be equally spaced
        self.interpolate_x()
        self.y = apply_relative_ctf(self.x, self.y, rel_ctf.x, rel_ctf.y)
    def fit_xrays(self):
        """
        Fits a model for cosmic rays (x rays) to the y data.

        Returns
        -------
        None.

        """
        self.xrays = xrays(self)
    @specstyle
    def plot_xrays(self):
        plt.plot(self.x, self.xrays, 'r-')
        plt.plot(self.x, self.y, 'k:')
    @change_y
    @log
    def remove_xrays(self):
        """
        Removes cosmic rays by subtracting the model stored in .xrays (if it exists).

        Returns
        -------
        None.

        """
        if hasattr(self, 'xrays'):
            self.y -= self.xrays
        else:
            pass
    def get_snr(self):
        """
        Approximates the signal-to-noise ratio of a Raman spectrum.

        Returns
        -------
        snr : double
            > Approximation for SNR.
        
        """
        return snr(self.y)
    @change_x
    @change_y
    @log
    def nm_to_wavenumber(self, laser_wavelength=532.):
        # The default is a frequency-doubled Ne:YAG laser with 523 nm
        self.x = wavelengths_to_wavenumbers(self.x, laser_wavelength)
        # The data will have to be flipped in general
        self.y = np.flip(self.y)
        self.x_label = 'Raman shift [rel. 1/cm]'
        
class RamanChada(RamanSpectrum):
    """
    Raman CHADA file with logging and saving to disc. Inherits from RamanSpectrum.
    """
    def __init__(self, source_path, raw=False,
             x_label='Raman shift [rel. 1/cm]', y_label='counts [1]'):
        """
        Parameters
        ----------
        source_path : str
            > Path to spectrum data file that is to be read.
            If extension is .cha, an existing CHADA file will be opened.
            If not, a native data file is imported and a CHADA file with the same name generated in the same directory.
            
        commit : str, optional
            > If specified, the data of the specific commit is loaded rather than the most recent. The default is [].
            
        x_label : str, optional
            > See RamanSpectrum. The default is 'Raman shift [rel. 1/cm]'.
            
        y_label : str, optional
            > See RamanSpectrum. The default is 'counts [1]'.

        Returns
        -------
        None.

        """
        # If file is not CHADA, create from native
        if os.path.splitext(source_path)[1] != '.cha':
            source_path = create_chada_from_native(source_path)
        self.x, self.y, self.meta, self.x_label, self.y_label = read_chada(source_path, raw=raw)
        self.file_path = source_path
        # Initialize log
        self.log = []
        # Save original state
        self.x0, self.y0 = self.x.copy(), self.y.copy()
        self.time = time.ctime()
    def show_log(self):
        """
        Shows the log.

        Returns
        -------
        None.

        """
        log = pd.DataFrame(data=self.log, columns=['time', 'method', 'arguments', 'keyword args'])
        print(log)
    def rewind(self, state):
        """
        Sets the object and data to a past state as listed in the log.

        Parameters
        ----------
        state : signed int
            > If positive, it sets the object to a past state as listed in the log.
            If negative, resets the object by -state steps.

        Returns
        -------
        None.

        """
        # Reset data
        self.x, self.y = self.x0.copy(), self.y0.copy()
        # get log up to state
        log = self.log[:state]
        # empty log (gets populated upon execution of methods)
        self.log = []
        for l in log:
            # Make function from log info
            # log lines are l = ['time', 'methodname', 'args', 'kwargs']
            func = getattr(self, l[1])
            func(*l[2], **l[3])
    @log
    def commit(self, commit_text="current"):
        """
        Makes a commit to the CHADA file by saving the current state to a new dataset within the existing HDF5 file.

        Parameters
        ----------
        commit : str, optional
            > Name as reference for commit. The default is 'current'.
            Cannot be 'raw', since the first commit after conversion (the raw data) cannot be edited.

        Returns
        -------
        None.

        """
        commit_chada(self, commit_text)
        # Initialize log
        self.log = []

def make_test_RamanChada():
    """
    Generates test spctrum
    Parameters
    ----------
    None.

    Returns
    -------
    RamanChada
        > Spectrum of a organic compound with several peaks as RamanChada.

    """
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, "testdata", "200218-17.wdf")
    return RamanChada(filename)

    
class SpectrumGroup:
    """
    Group of spectra for comparison and multivariate analysis
    """
    def __init__(self, spectra=[]):
        """
        Parameters
        ----------
        spectra : RamanSpectrum, optional
            > Raman spectra to be contained upon initialization. The default is [].

        Returns
        -------
        None.

        """
        self.spectra = []
        self.data_labels = []
        for spectrum in spectra:
            self.add(spectrum)
    def __repr__(self):
        info1 = f'{self.__class__.__name__} containing {len(self.spectra)} objects.'
        info2 = f'Object type(s): {set([s.__class__.__name__ for s in self.spectra])}'
        info = info1 + '\n' + info2
        return info
    def add(self, spectrum):
        """
        Adds a RamanSpectrum to the Group.

        Parameters
        ----------
        spectrum : RamanSpectrum

        Returns
        -------
        None.

        """
        self.spectra.append( deepcopy(spectrum) )
        self.x_label, self.y_label = spectrum.x_label, spectrum.y_label
        if hasattr(spectrum, 'meta'):
            self.data_labels.append(spectrum.meta['Original file'])
        else:
            self.data_labels.append(len(self.data_labels))
    @specstyle
    def plot(self, leg=True):
        """
        Plot all contained spectra into a single graph.

        Parameters
        ----------
        leg : bool, optional
            > True if a legend should be shown. The default is True.
            If unspecified, legend labelsare set to file names.

        Returns
        -------
        None.

        """
        for s, l in zip(self.spectra, self.data_labels):
            plt.plot(s.x, s.y, label=l)
        if leg:
            plt.legend()
#    - add_labels
#    - add_targets
    def to_array(self, x_increment=.5):
        """
        Interpolate all spectra to a common x axis and Export as Numpy array.

        Parameters
        ----------
        x_increment : double, optional
            > Spectral increment (resolution) of the common x axis. The default is 0.5.

        Returns
        -------
        np.array
            > Matrix containing all spectra in Group as lines.

        """
        # Boundaries of intersection
        lo = max([s.x.min() for s in self.spectra])
        hi = min([s.x.max() for s in self.spectra])
        # Make common x
        x = np.arange(lo, hi, x_increment)
        # for all
        y = []
        for s in self.spectra:
            # interpolate to x
            f_inter = interp1d(s.x, s.y, kind="quadratic")
            y.append(f_inter(x))
        return x, np.array(y)
    def process(self, method, *args, **kwargs):
        """
        Applies a specified method to separately to each spectrum in the group.
        Example:
            
            G.process('x_crop', 500, 1800)

        Parameters
        ----------
        method : str
            > Name of method to be applied
            
        *args
            > Non-keyword agruments for method. Refer to method documentation.
            
        **kwargs : TYPE
            > Keyword agruments for method. Refer to method documentation.

        Returns
        -------
        None.

        """
        # for indexed spectra
        if 'index' in kwargs:
            index = kwargs.pop('index', 0)
            spectra = [ self.spectra[i] for i in index ]
        else:
            spectra = self.spectra
        for s in spectra:
            # search for method in class object
            try:
                func = getattr(s, method)
                # Apply method with args
                func(*args, **kwargs)
            except Exception as err: raise err
    def make_mtf_calibration(self):
        """
        Extract x rays from grouped spectra and generate a model from the average MTF in Fourier space.
        This is suitable if the Group consists of a time series, which is likely to contain one or more x rays.

        Returns
        -------
        RamanMTF
            > MTF model in Fourier space.

        """
        xray_list = []
        for s in self.spectra:
            xray_list.extend(extract_xrays(s))
        print(str(len(xray_list)) + ' xrays found in Group')
        k, mtf = raman_mtf_from_psfs(xray_list)
        print('calculated MTF with amlitude = ' + str(mtf[-1]) + ' at full Nyquist.')
        mtf_data = pd.DataFrame()
        mtf_data['Nyquist frequency'] = k
        mtf_data['amplitude'] = mtf
        return RamanMTF(mtf_data)
    def round_robin(self, ref_pos, fitmethod='voigt', interval_width=2):
        """
        Determines the x position of a list of specified peaks.

        Parameters
        ----------
        fitmethod : str
            > Name of peak fit model to be applied
            
        interval_width : int
            > Interval in multiples of FWHM around the peak to be used for peak fitting.  

        Returns
        -------
        DataFrame
            > *DataFrame* containing standard and fitted peak positions.
        """
        lines, labels = [], []
        for s in self.spectra:
            # Fit peaks for each spectrum
            s.peaks(fitmethod=fitmethod, interval_width=interval_width)
            if len(s.bands) > 0:
                # Select peak closest to indicated reference pos
                pos = s.bands.position.to_numpy()
                peak_line = np.argmin(np.abs(pos-ref_pos))
                # Combnine DataFrame from target peaks
                lines.append(s.bands.loc[peak_line].to_dict())
                labels.append(s.meta['Original file'])
        return pd.DataFrame(lines, index=labels)
    def generate_labels(self, pivot_string=None, pos=0, numeric=True, length=False):
        # Make list of filenames without path and ext.
        data_labels = [os.path.splitext(self.data_labels)[0] for s in self.spectra]
        return labels_from_filenames(data_labels, pivot_string=pivot_string, pos=pos, numeric=numeric, length=length)


class RamanGroup():
    """
    Group of Raman spectra for comparison and multivariate analysis.
    Contains the spectra in form of a DataFrame in the *.data* attribute
    with spectra as rows and Raman shifts as columns. The column names are
    Raman shifts as *float*, while the index is the file basename.
    """
    def __init__(self, spectra, interpolate=True):
        """
        Parameters
        ----------
        spectra : list of *RamanChada* or *RamanSpectrum* objects
            > Spectra to be included upon initialization.
            Example:
            
                G = RamanGroup( [s1, s2, s3] )
            
        interpolate : bool
            > If True, Raman shifts are interpolated to increments of 1 1/cm.
            If False, all x axes will be interpolated to the x axis of the first spectrum added.

        Returns
        -------
        None.
        """
        first_spectrum = spectra[0]
        if interpolate:
            first_spectrum.interpolate_x()
        self.x_label, self.y_label = first_spectrum.x_label, first_spectrum.y_label
        S = spectrum_to_frame(first_spectrum)
        self.data = pd.DataFrame(S)
        if len(spectra) > 1:
            self.add(spectra[1:])
            
    def add_one(self, spectrum, use_new_x=False):
        """
        Adds a single spectrum.
        Parameters
        ----------
        spectrum : *RamanChada* or *RamanSpectrum* object
            > Spectrum to be added.
            Example:
            
                G.add_one(s4)
            
        use_new_x : bool
            > If True, the existing x axes will be interpolated to the newly added spectrum.
            If False, the x axis will be interpolated on the existing x axes. The default is False.

        Returns
        -------
        None.
        """
        # Convert added spectrum to DataFrame with x values as rows
        S = spectrum_to_frame(spectrum).T
        # Merge existing self.data with S while using the 'outer' union with both sets of x values
        merged = self.data.T.merge(S, left_index=True, right_index=True, how='outer')
        # Interpolate missing values and reindex on original x axis. Fill NaNs with zeros.
        if use_new_x:
            new_index = S.index
        else:
            new_index = self.data.T.index
        merged = merged.interpolate('quadratic').reindex(new_index).fillna(0)
        self.data = merged.T
    
    def add(self, spectra):
        """
        Adds a list of spectra.
        Parameters
        ----------
        spectra : list of *RamanChada* or *RamanSpectrum* objects
            > Spectra to be added.
            Example:
            
                G.add( [s5, s6] )
            
        Returns
        -------
        None.
        """
        [ self.add_one(s) for s in spectra ]
        
    def set_targets(self, target_dict):
        """
        Adds targets for plottting prediction model training.
        Parameters
        ----------
        target_dict : dict
            > Targets to be added to the group as dict.
            Keys are arbitrary target names, while values must be lists
            or 1-dim arrays with a lenght matching the number of spectra in the *RamanGroup*,
            i.e. `len(target_dict) = len(G.data)`.
            Target values can be strings (for classification) or numbers (for regression).
            The targets are stored in *RamanGroup.targets*, a *DataFrame*.
            Note that multiple targets can be added.
            
        Returns
        -------
        None.
        """
        self.targets = pd.DataFrame(index=self.data.index, data=target_dict)
        
    def extract(self, no_of_spectrum):
        """
        Returns a specified single spectrum from the *RamanGroup*.
        Parameters
        ----------
        no_of_spectrum : int
            > Index of the spectrum to be extracted.
            
        Returns
        -------
        RamanChada
            > Extracted spectrum
        """
        return line_to_spectrum(self.data, no_of_spectrum)

    def clone(self):
        """
        Returns a deep copy of a *RamanGroup*.
        Parameters
        ----------
        None.
            
        Returns
        -------
        RamanGroup
            > Exact copy of *self*.
        """
        return deepcopy(self)
    
    def process(self, method, *args, **kwargs):
        """
        Applies a single pre-processing step to a *RamanGroup*.
        Uses the function *ramanchada.models.process_DF*.
        Parameters
        ----------
        method : str
            > Name of *RamanChada* method to be applied.
            For details, refer to *ramanchada.RamanChada*.
            
        *args : str of number
            > Non-keyword parameters for method

        **kwargs
            > Keyword parameters for method   

        Returns
        -------
        None.
        """
        self.data = process_DF(self.data, method, *args, **kwargs)
        
    @property
    def x(self):
        """
        np.array
            > Common x axis of *RamanGroup* as 1D array
        """
        return np.array(self.data.columns)
    
    @property
    def y(self):
        """
        np.array
            > Y values of *RamanGroup* as array with dimensions [n_spectra, n_channels]
        """
        return self.data.to_numpy()
        
    def __repr__(self):
        info = f'{self.__class__.__name__} containing {len(self.data)} spectra.'
        if hasattr(self, 'targets'):
            info += '\n' + f'{len(self.targets)} added'
        return info

    def plot(self, target=None, xrange = [1e-9, 1e9], legend=False):
        """
        Plots spectra of a *RamanGroup* in a single axis. 
        Parameters
        ----------
        target : str
            > Name of target to be used for hue. If given, spectra with a
            common target will be represented by their mean spectrum, with
            a spread corresponding to the standard deviation at each Raman shift.
            This is particularly useful if a large number of spectra are plotted.
            The default is None.

        xrange : list of double [x_min, x_max]
            > x interval to be plotted. The default is [1e-9, 1e9] (all).

        legend : bool
            > If True, a legend will be shown. The default is False. 

        Returns
        -------
        None.
        """
        plot_style.use('bmh')
        plt.figure(figsize=[8,4])
        if target:
            #target = 'times'
            arrays = [self.targets[target], self.targets.index]
            DF = self.data.copy()
            DF.index = pd.MultiIndex.from_arrays(arrays, names=(target, 'file'))
            sns.lineplot(data=DF.T[xrange[0]:xrange[1]], ci='sd', n_boot=5, dashes=False)
        else:
            sns.lineplot(data=self.data.T[xrange[0]:xrange[1]], dashes=False, legend=legend)
        plt.ylabel(self.y_label)
        plt.show()


class RamanCalibration(Curve):
    """
    Object containing a Raman x axis calibration.
    """
    def __init__(self, data, poly_degree=3, interpolate=False):
        """
        Parameters
        ----------
        data : DataFrame
            > DataFrame with two columns, the first containing the x positions and the second containing the shifts at these positions.
            
        poly_degree : int, optional
            > Degree of polynomial model fitted to the data points. The default is 3.
            
        interpolate : bool, optional
            > If True, values of the polynomial model which are outside of the data range used for calibration are interpolated.
            If False, all shifts outside the data range are set to the boundary values.
            The default is False.

        Returns
        -------
        None.

        """
        super().__init__(data, data.columns[0], data.columns[1])
        # fit shift vector
        # If specified, use 1d linear interpolation
        if len(self.x) == 0:
            self.interp_x = np.zeros_like
        else:
            if interpolate:
                self.interp_x = np.poly1d(np.polyfit(self.x, self.y, poly_degree))
            else:
                # Only apply shift within x limits of calibration data
                # Set shifts outside the limits to boundary values
                self.interp_x = interpolation_within_bounds(self.x, self.y, poly_degree)
            self.poly_degree = poly_degree
            self.interpolate = interpolate
        self.x_offset_value = 0
    def show(self):
        """
        Plots the calibration data points and the associated model.

        Returns
        -------
        None.

        """
        plt.figure()
        test_x = np.linspace(0, self.x.max(), 100)
        plt.plot(test_x, self.interp_x(test_x) + self.x_offset_value)
        plt.plot(self.x, self.y, 'ko')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()
    def save(self, file_path):
        calibration_metadata = {'Calibration time': self.time, 'Polynomial order': self.poly_degree,
            'Interpolate': self.interpolate, 'x offset': self.x_offset_value}
        write_new_chada(file_path, self.x, self.y, calibration_metadata)


class RamanMTF(Curve):
    """
    Object containing an MTF model in Fourier space. The reciprocal coordinate is 1/pixels.
    """
    def __init__(self, data):
        super().__init__(data, 'Nyquist frequency', 'amplitude')
        # k + CTF come as 1024 px vectors.


class RamanCTF(Curve):
    """
    Object containing an MTF model in Fourier space. The reciprocal coordinate is 1/x units.
    """
    def __init__(self, data):
        super().__init__(data, 'spatial frequency', 'amplitude')
        # k + CTF come as 1024 px vectors.


class MetaData(dict):
    """
    Object containing meta data as a dict.
    """
    pass
    # Can be loaded or generated as dict, independently from data
#    - from_file
#    - clean


def spectrum_to_frame(spectrum):
    """
    Converts a spectrum to a *DataFrame* with a single row.
    Parameters
    ----------
    spectrum : Spectrum, RamanSpectrum or RamanChada
        > Spectrum to be converted
    
    Returns
    -------
    DataFrame
        > DataFrame with a single row.
        For details, see *ramanchada.classes.RamanGroup*
    """
    if hasattr(spectrum, 'meta'):
        column_name = spectrum.meta['Original file']
    else:
        column_name = 'intensity'
    D = pd.DataFrame({column_name: spectrum.y})
    D.index = spectrum.x
    D.index.name = spectrum.x_label
    D.attrs = {'y_label': spectrum.y_label}
    return D.T

def spectrum_to_series(spectrum):
    """
    Converts a spectrum to a pandas *Series*.
    Parameters
    ----------
    spectrum : Spectrum, RamanSpectrum or RamanChada
        > Spectrum to be converted
    
    Returns
    -------
    pandas.Series
        > Series with Raman shifts as index.
    """
    S = pd.Series(data=spectrum.y, name=spectrum.meta['Original file'])
    S.index = spectrum.x
    S.index.name = spectrum.x_label
    return S

def line_to_spectrum(DF, no_line):
    """
    Converts the specified row of a *DataFrame* to a *Spectrum*.
    Parameters
    ----------
    DF : DataFrame
        > *pandas.DataFrame* structured as described in *ramanchada.classes.RamanGroup*
    
    no_line : int
        > Row index to be converted 
    
    Returns
    -------
    ramanchada.classes.Spectrum
    """
    S = DF.iloc[no_line]
    S.name = 'y'
    S = S.to_frame()
    S['x'] = S.index
    return Spectrum(S, x_column_name='x', y_column_name='y')

def process_DF(DF, method, *args, **kwargs):
    """
    Applies a single pre-processing step to each row of a *DataFrame*.
    Parameters
    ----------
    method : str
        > Name of *RamanChada* method to be applied.
        For details, refer to *ramanchada.classes.RamanChada*.
        
    *args : str of number
        > Non-keyword parameters for method

    **kwargs
        > Keyword parameters for method   

    Returns
    -------
    *pandas.DataFrame*
        > Processed *DataFrame*
    """
    S = []
    for row in range(len(DF)):
        s = line_to_spectrum(DF, row)
        func = getattr(s, method)
        func(*args, **kwargs)
        S.append(s.y)
    Y = np.array(S)
    DF_proc = pd.DataFrame(Y, columns = s.x)
    DF_proc.columns.name = DF.columns.name
    DF_proc.index = DF.index
    return DF_proc

