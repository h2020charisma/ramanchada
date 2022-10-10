# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Feb 26, 2021
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

# Python libraries
from ramanchada.classes import RamanChada, SpectrumGroup,  make_test_RamanChada
from ramanchada.analysis.peaks import voigt_fit, find_spectrum_peaks, double_voigt_fit, gauss_lorentz_fit, pearson4_fit, beta_profile_fit, lorentz_fit

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Int64Index as NumericIndex


def peak_fit_all(path):
    """
    Fitting function : voigt default parameters,
    Area 0 - 3500 for all peaks
    """


    RR = pd.DataFrame()
    RRR = pd.DataFrame()
    #path = os.path.dirname(file)
    for fitmethod in ['voigt', 'par','lorentz']:
        for spectra in os.listdir(path):
            filename = os.fsdecode(spectra)
            if filename.endswith(".txt"):
                print(filename)
                f = os.path.join(path,filename)
                spec_NTUA=RamanChada(f)
                spec_NTUA.x_crop(0,3500)
                spec_NTUA.plot()
                spec_NTUA.normalize('area')
                spec_NTUA.peaks(fitmethod=fitmethod,show=True)
                print(spec_NTUA.bands)
                RR['Peak positions'] = spec_NTUA.bands['position']
                RR[str(fitmethod)+' Intensity'] = spec_NTUA.bands['intensity']
                RR[str(fitmethod)+' Prominence'] = spec_NTUA.bands['prominence']
                RR[str(fitmethod)+' fitted position'] = spec_NTUA.bands[str(fitmethod)+' fitted position']
                RR[str(fitmethod)+' fitted FWHM'] = spec_NTUA.bands[str(fitmethod)+' fitted FWHM']
                RR['FWHM'] = spec_NTUA.bands['FWHM']
                print(RR)
                RRR = RRR.append(RR, ignore_index = True)
    global RRR2
    RRR.dropna(inplace=True)
    RRR2= RRR.groupby(['Peak positions'],as_index=False).mean()
    print()
    print('Remove of NaN values and estimation of average values per peak')
    RRR2
    return RRR2




def peak_fit_all_full(path):
    """
    ##Fitting function : voigt default parameters,
    ##Area 0 - 3500 for all peaks
    """
    RR = pd.DataFrame()
    RRR = pd.DataFrame()
    #path = os.path.dirname(file)
    for fitmethod in ['voigt', 'par','lorentz']:
        for spectra in os.listdir(path):
            filename = os.fsdecode(spectra)
            if filename.endswith(".txt"):
                print(filename)
                f = os.path.join(path,filename)
                spec_NTUA=RamanChada(f)
                # Correction of fluorescent background
                print('Correction of fluorescent background')
                print('Fit baseline model using the SNIP algorithm and remove from data')
                spec_NTUA.fit_baseline(method='snip') # Fit baseline model using the SNIP algorithm and remove from data
                spec_NTUA.remove_baseline()
                print('Reset data. Fit baseline model using the ALS algorithm and remove from data')
                spec_NTUA.rewind(0) # Reset data. Fit baseline model using the ALS algorithm and remove from data
                spec_NTUA.fit_baseline(method='als')
                spec_NTUA.remove_baseline()
                # Correction of cosmic rays
                print('Correction of cosmic rays')
                print('Fit x ray model')
                spec_NTUA.fit_xrays() # Fit x ray model
                plt.figure()
                plt.plot(spec_NTUA.y, label='raw data')
                plt.plot(spec_NTUA.xrays, label='x ray model')
                plt.legend()
                plt.show()
                print('Subtract x ray model and plot corrected spectrum')
                spec_NTUA.remove_xrays() #  Subtract x ray model and plot corrected spectrum
                spec_NTUA.plot()
                # Noise reduction with minimal information loss
                print('Noise reduction with minimal information loss')
                G = SpectrumGroup([spec_NTUA])
                print('Apply Savitzky-Golay smoothing filter')
                spec_NTUA.smooth(method='sg')
                G.add(spec_NTUA)
                G.process('x_crop', 0, 1500)
                G.plot()
                # Correction of wavenumber shift - x calibration
                print('Correction of wavenumber shift - x calibration')
                # Choose Neon calibration file, needs to be defined properly
                #cwd = os.getcwd()
                #NEON_REF= os.path.join(cwd,NEON_file)
                n='Neon_S120_x20_7000msx15.cha'
                NEON_REF=RamanChada(n)
                NEON_REF.x_crop(0,3500)
                NEON_REF.plot()
                N = SpectrumGroup([NEON_REF, spec_NTUA])
                N.process('x_crop', 0, 3200)
                N.process('normalize')
                N.plot()
                print('Fit calibration by peak position comparison, using a Voigt distribution as peak model')
                neon_cal = spec_NTUA.make_x_calibration(NEON_REF)
                neon_cal
                print('Plot x calibration data (circles) and model (5th order polynomial, red).')
                print('RS correction: Wavenumber-dependent x shift applied by the calibration.')
                print('Note that shifts outside the interval of mesured peaks are set to the boundary shifts rather than being interpolated')
                neon_cal.show()
                print('Calibrate the target spectrum using the generated calibration')
                spec_NTUA.calibrate(neon_cal)
                print('Group calibrated spectrum with original data to see result.')
                N.add(spec_NTUA)
                N.process('x_crop', 1000, 3000)
                N.process('normalize', 'area')
                N.plot()
                print('red - Reference spectrum')
                print('blue - original Target spectrum')
                print('purple - Target calibrated to Reference')

                # Extra added features
                #spec_NTUA.x_crop(0,3500)
                #spec_NTUA.plot()
                spec_NTUA.normalize('minmax')
                spec_NTUA.peaks(fitmethod=fitmethod,show=True)
                print(spec_NTUA.bands)
                RR['Peak positions'] = spec_NTUA.bands['position']
                RR[str(fitmethod)+' Intensity'] = spec_NTUA.bands['intensity']
                RR[str(fitmethod)+' Prominence'] = spec_NTUA.bands['prominence']
                RR[str(fitmethod)+' fitted position'] = spec_NTUA.bands[str(fitmethod)+' fitted position']
                RR[str(fitmethod)+' fitted FWHM'] = spec_NTUA.bands[str(fitmethod)+' fitted FWHM']
                RR['FWHM'] = spec_NTUA.bands['FWHM']
                print(RR)
                RRR = RRR.append(RR, ignore_index = True)

    global RRR2
    RRR2= pd.DataFrame()
    RRR1= RRR.dropna()
    RRR2= RRR1.groupby(['Peak positions'],as_index=False).mean()
    print()
    print('Remove of NaN values and estimation of average values per peak')
    RRR2
    return RRR2






def peak_fit_EP(path):
    """
    Fitting function : voigt default parameters,
    Area 0 - 3500 for 5 reference peaks from European Pharmacopoeia
    """
    # Crop to specified range
    #l = lims(x, x_min, x_max)
    #x, y = l(x), l(y)
    # Filter + minmax normalization are important!
    ps_peak_pos = [
    620.9,
    1001.4,
    1031.8,
    1602.3,
    3054.3]

    RR = pd.DataFrame()
    RRR = pd.DataFrame()
    #path = os.path.dirname(file)
    for fitmethod in ['voigt', 'par','lorentz']:
        for spectra in os.listdir(path):
            filename = os.fsdecode(spectra)
            if filename.endswith(".txt"):
                print(filename)
                f = os.path.join(path,filename)
                spec_NTUA=RamanChada(f)
                spec_NTUA.x_crop(0,3500)
                spec_NTUA.plot()
                spec_NTUA.normalize('area')
                spec_NTUA.peaks(fitmethod=fitmethod,show=True)
                ps_peak_pos.sort(reverse=True)
                spec_NTUA.bands['PS position'] = np.nan
                for pos in ps_peak_pos:
                    index = np.argmin(np.abs(spec_NTUA.bands['position']-pos))
                    spec_NTUA.bands['PS position'][index] = pos
                spec_NTUA.bands
                RR['Peak reference positions'] = spec_NTUA.bands['PS position']
                RR[str(fitmethod)+' Intensity'] = spec_NTUA.bands['intensity']
                RR[str(fitmethod)+' Prominence'] = spec_NTUA.bands['prominence']
                RR[str(fitmethod)+' fitted position'] = spec_NTUA.bands[str(fitmethod)+' fitted position']
                RR[str(fitmethod)+' fitted FWHM'] = spec_NTUA.bands[str(fitmethod)+' fitted FWHM']
                RR['FWHM'] = spec_NTUA.bands['FWHM']
                print(RR)
                RRR = RRR.append(RR, ignore_index = True)

    global RRR2
    RRR.dropna(inplace=True)
    RRR2= RRR.groupby(['Peak reference positions'],as_index=False).mean()
    print()
    print('Remove of NaN values and estimation of average values per peak')
    RRR2
    return RRR2



def peak_fit_EP_full(path):

    ##Fitting function : voigt default parameters,
    ## Area 0 - 3500 for 5 reference peaks from European Pharmacopoeia

    ps_peak_pos = [
    620.9,
    1001.4,
    1031.8,
    1602.3,
    3054.3]

    RR = pd.DataFrame()
    RRR = pd.DataFrame()
    #path = os.path.dirname(file)
    for fitmethod in ['voigt', 'par','lorentz']:
        for spectra in os.listdir(path):
            filename = os.fsdecode(spectra)
            if filename.endswith(".txt"):
                print(filename)
                f = os.path.join(path,filename)
                spec_NTUA=RamanChada(f)
                # Correction of fluorescent background
                print('Correction of fluorescent background')
                print('Fit baseline model using the SNIP algorithm and remove from data')
                spec_NTUA.fit_baseline(method='snip') # Fit baseline model using the SNIP algorithm and remove from data
                spec_NTUA.remove_baseline()
                print('Reset data. Fit baseline model using the ALS algorithm and remove from data')
                spec_NTUA.rewind(0) # Reset data. Fit baseline model using the ALS algorithm and remove from data
                spec_NTUA.fit_baseline(method='als')
                spec_NTUA.remove_baseline()
                # Correction of cosmic rays
                print('Correction of cosmic rays')
                print('Fit x ray model')
                spec_NTUA.fit_xrays() # Fit x ray model
                plt.figure()
                plt.plot(spec_NTUA.y, label='raw data')
                plt.plot(spec_NTUA.xrays, label='x ray model')
                plt.legend()
                plt.show()
                print('Subtract x ray model and plot corrected spectrum')
                spec_NTUA.remove_xrays() #  Subtract x ray model and plot corrected spectrum
                spec_NTUA.plot()
                # Noise reduction with minimal information loss
                print('Noise reduction with minimal information loss')
                G = SpectrumGroup([spec_NTUA])
                print('Apply Savitzky-Golay smoothing filter')
                spec_NTUA.smooth(method='sg')
                G.add(spec_NTUA)
                G.process('x_crop', 0, 1500)
                G.plot()
                # Correction of wavenumber shift - x calibration
                print('Correction of wavenumber shift - x calibration')
                # Choose Neon calibration file, needs to be defined properly
                #cwd = os.getcwd()
                #NEON_REF= os.path.join(cwd,NEON_file)
                n='Neon_S120_x20_7000msx15.cha'
                NEON_REF=RamanChada(n)
                NEON_REF.x_crop(0,3500)
                NEON_REF.plot()

                N = SpectrumGroup([NEON_REF, spec_NTUA])
                N.process('x_crop', 0, 3200)
                N.process('normalize')
                N.plot()
                print('Fit calibration by peak position comparison, using a Voigt distribution as peak model')
                neon_cal = spec_NTUA.make_x_calibration(NEON_REF)
                neon_cal
                #Plot x calibration data (circles) and model (5th order polynomial, red).
                #RS correction: Wavenumber-dependent x shift applied by the calibration.
                #Note that shifts outside the interval of mesured peaks are set to the boundary shifts rather than being interpolated
                neon_cal.show()
                print('Calibrate the target spectrum using the generated calibration')
                spec_NTUA.calibrate(neon_cal)
                print('Group calibrated spectrum with original data to see result.')
                N.add(spec_NTUA)
                N.process('x_crop', 1000, 3000)
                N.process('normalize', 'area')
                N.plot()
                print('red - Reference spectrum')
                print('blue - original Target spectrum')
                print('purple - Target calibrated to Reference')


                # Extra added features
                spec_NTUA.x_crop(0,3500)
                spec_NTUA.plot()
                spec_NTUA.normalize('minmax')
                spec_NTUA.peaks(fitmethod=fitmethod,show=True)
                ps_peak_pos.sort(reverse=True)
                spec_NTUA.bands['PS position'] = np.nan
                for pos in ps_peak_pos:
                    index = np.argmin(np.abs(spec_NTUA.bands['position']-pos))
                    spec_NTUA.bands['PS position'][index] = pos
                spec_NTUA.bands
                RR['Peak reference positions'] = spec_NTUA.bands['PS position']
                RR[str(fitmethod)+' Intensity'] = spec_NTUA.bands['intensity']
                RR[str(fitmethod)+' Prominence'] = spec_NTUA.bands['prominence']
                RR[str(fitmethod)+' fitted position'] = spec_NTUA.bands[str(fitmethod)+' fitted position']
                RR[str(fitmethod)+' fitted FWHM'] = spec_NTUA.bands[str(fitmethod)+' fitted FWHM']
                RR['FWHM'] = spec_NTUA.bands['FWHM']
                print(RR)
                RRR = RRR.append(RR, ignore_index = True)

    global RRR2
    RRR2= pd.DataFrame()
    RRR1= RRR.dropna()
    RRR2= RRR1.groupby(['Peak reference positions'],as_index=False).mean()
    print()
    print('Remove of NaN values and estimation of average values per peak')
    RRR2
    return RRR2







def peak_fit_EP_dist(path):
    """
    Fitting function : voigt default parameters,
    Distinct fitting area for each 5 reference peaks from European Pharmacopoeia within 0 - 3500 range
    """

    ps_peak_pos = [
    620.9,
    1001.4,
    1031.8,
    1602.3,
    3054.3]

    RR = pd.DataFrame()
    RRR = pd.DataFrame()
    #path = os.path.dirname(file)
    ps_peak_pos.sort(reverse=True)
    for fitmethod in ['voigt', 'par','lorentz']:
        for pos in ps_peak_pos:
            for spectra in os.listdir(path):
                filename = os.fsdecode(spectra)
                if filename.endswith(".txt"):
                    print(filename)
                    f = os.path.join(path,filename)
                    spec_NTUA=RamanChada(f)
                    spec_NTUA.x_crop(pos-70,pos+70)
                    spec_NTUA.plot()
                    spec_NTUA.normalize('area')
                    spec_NTUA.peaks(fitmethod=fitmethod,show=True)
                    spec_NTUA.bands['PS position'] = np.nan
                    index = np.argmin(np.abs(spec_NTUA.bands['position']-pos))
                    spec_NTUA.bands['PS position'][index] = pos
                    spec_NTUA.bands
                    RR['Peak reference positions'] = spec_NTUA.bands['PS position']
                    RR[str(fitmethod)+' Intensity'] = spec_NTUA.bands['intensity']
                    RR[str(fitmethod)+' Prominence'] = spec_NTUA.bands['prominence']
                    RR[str(fitmethod)+' fitted position'] = spec_NTUA.bands[str(fitmethod)+' fitted position']
                    RR[str(fitmethod)+' fitted FWHM'] = spec_NTUA.bands[str(fitmethod)+' fitted FWHM']
                    RR['FWHM'] = spec_NTUA.bands['FWHM']
                    print(RR)
                    RRR = RRR.append(RR, ignore_index = True)

    global RRR2
    RRR.dropna(inplace=True)
    RRR2= RRR.groupby(['Peak reference positions'],as_index=False).mean()
    print()
    print('Remove of NaN values and estimation of average values per peak')
    RRR2
    return RRR2


def peak_fit_EP_dist_full(path):
    """
    Fitting function : voigt default parameters,
    Distinct fitting area for each 5 reference peaks from European Pharmacopoeia within 0 - 3500 range
    """

    ps_peak_pos = [
    620.9,
    1001.4,
    1031.8,
    1602.3,
    3054.3]

    RR = pd.DataFrame()
    RRR = pd.DataFrame()
    #path = os.path.dirname(file)
    ps_peak_pos.sort(reverse=True)
    for fitmethod in ['voigt', 'par','lorentz']:
        for pos in ps_peak_pos:
            for spectra in os.listdir(path):
                filename = os.fsdecode(spectra)
                if filename.endswith(".txt"):
                    print(filename)
                    f = os.path.join(path,filename)
                    spec_NTUA=RamanChada(f)
                    # Correction of fluorescent background
                    print('Correction of fluorescent background')
                    print('Fit baseline model using the SNIP algorithm and remove from data')
                    spec_NTUA.fit_baseline(method='snip') # Fit baseline model using the SNIP algorithm and remove from data
                    spec_NTUA.remove_baseline()
                    print('Reset data. Fit baseline model using the ALS algorithm and remove from data')
                    spec_NTUA.rewind(0) # Reset data. Fit baseline model using the ALS algorithm and remove from data
                    spec_NTUA.fit_baseline(method='als')
                    spec_NTUA.remove_baseline()
                    # Correction of cosmic rays
                    print('Correction of cosmic rays')
                    print('Fit x ray model')
                    spec_NTUA.fit_xrays() # Fit x ray model
                    plt.figure()
                    plt.plot(spec_NTUA.y, label='raw data')
                    plt.plot(spec_NTUA.xrays, label='x ray model')
                    plt.legend()
                    plt.show()
                    print('Subtract x ray model and plot corrected spectrum')
                    spec_NTUA.remove_xrays() #  Subtract x ray model and plot corrected spectrum
                    spec_NTUA.plot()
                    # Noise reduction with minimal information loss
                    print('Noise reduction with minimal information loss')
                    G = SpectrumGroup([spec_NTUA])
                    print('Apply Savitzky-Golay smoothing filter')
                    spec_NTUA.smooth(method='sg')
                    G.add(spec_NTUA)
                    G.process('x_crop', 0, 1500)
                    G.plot()
                    # Correction of wavenumber shift - x calibration
                    print('Correction of wavenumber shift - x calibration')
                    # Choose Neon calibration file, needs to be defined properly
                    #cwd = os.getcwd()
                    #NEON_REF= os.path.join(cwd,NEON_file)
                    n='Neon_S120_x20_7000msx15.cha'
                    NEON_REF=RamanChada(n)
                    NEON_REF.x_crop(0,3500)
                    NEON_REF.plot()

                    N = SpectrumGroup([NEON_REF, spec_NTUA])
                    N.process('x_crop', 0, 3200)
                    N.process('normalize')
                    N.plot()
                    print('Fit calibration by peak position comparison, using a Voigt distribution as peak model')
                    neon_cal = spec_NTUA.make_x_calibration(NEON_REF)
                    neon_cal
                    #Plot x calibration data (circles) and model (5th order polynomial, red).
                    #RS correction: Wavenumber-dependent x shift applied by the calibration.
                    #Note that shifts outside the interval of mesured peaks are set to the boundary shifts rather than being interpolated
                    neon_cal.show()
                    print('Calibrate the target spectrum using the generated calibration')
                    spec_NTUA.calibrate(neon_cal)
                    print('Group calibrated spectrum with original data to see result.')
                    N.add(spec_NTUA)
                    N.process('x_crop', 1000, 3000)
                    N.process('normalize', 'area')
                    N.plot()
                    print('red - Reference spectrum')
                    print('blue - original Target spectrum')
                    print('purple - Target calibrated to Reference')

                    # Extra added features
                    spec_NTUA.x_crop(pos-70,pos+70)
                    spec_NTUA.plot()
                    spec_NTUA.normalize('minmax')
                    spec_NTUA.peaks(fitmethod=fitmethod,show=True, interval_width=2)
                    spec_NTUA.bands['PS position'] = np.nan
                    index = np.argmin(np.abs(spec_NTUA.bands['position']-pos))
                    spec_NTUA.bands['PS position'][index] = pos
                    spec_NTUA.bands
                    RR['Peak reference positions'] = spec_NTUA.bands['PS position']
                    RR[str(fitmethod)+' Intensity'] = spec_NTUA.bands['intensity']
                    RR[str(fitmethod)+' Prominence'] = spec_NTUA.bands['prominence']
                    RR[str(fitmethod)+' fitted position'] = spec_NTUA.bands[str(fitmethod)+' fitted position']
                    RR[str(fitmethod)+' fitted FWHM'] = spec_NTUA.bands[str(fitmethod)+' fitted FWHM']
                    RR['FWHM'] = spec_NTUA.bands['FWHM']
                    print(RR)
                    RRR = RRR.append(RR, ignore_index = True)

    global RRR2
    RRR2= pd.DataFrame()
    RRR1= RRR.dropna()
    RRR2= RRR1.groupby(['Peak reference positions'],as_index=False).mean()
    print()
    print('Remove of NaN values and estimation of average values per peak')
    RRR2
    return RRR2



def export_table(x):
    RRR2.to_excel(x)
    print('DataFrame is written to Excel File successfully.')


