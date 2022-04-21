from ramanchada.classes import RamanChada, SpectrumGroup
import pandas as pd
import numpy as np

#Based on JT notebook "Calibration Protocol"
#at the moment it just checks if the code is running without error
#tbd - asserts to check bands / peaks fitting outcomes
def test_calibration(file = 'test/calibration/NEON_GlacX785_BWTek_Probe_P0_850msx5_raw.csv'):
    Neon = RamanChada(file)
    # Set neon spectrum to pixel numbers
    Neon.reset_x()
    # Find peak positions of neon peaks
    Neon.peaks(fitmethod='par', interval_width=0.5, show=False, prominence=.002, sort_by='position')
    #print(Neon.bands) tbd assert
    neon_x_axisWL = Neon.make_x_axis({
    2: 966.542,
    8: 937.331,
    16: 914.867,
    18: 891.95,
    23: 878.062,
    29: 863.465,
    34: 849.536,
    40: 837.761,
    43: 830.032,
    45: 813.641,
    48: 794.318}, x_unit='wavelength [nm]', column='par fitted position', order=5)
    Neon.assign_x(neon_x_axisWL)
    assert 1856 == len(Neon.y)
    Neon.peaks(prominence=.002, sort_by='position')
    #print(Neon.bands)
    Neon.peaks(fitmethod='par', interval_width=0.1, show=False, prominence=.002, sort_by='position')
    #Neon.bands