from copy import deepcopy
import numpy as np
import pandas as pd

from .classes import process_DF, RamanGroup, Spectrum, spectrum_to_frame

class RamanProcessor():

    def __init__(self, **steps):
        if steps == {}:
            steps = {'fit_baseline': ['snip'], 'normalize': ['snv']}
        self.steps = steps

    def fit(self, df, params=None):
        self.x = np.array(df.columns)
        return self

    def transform(self, data):
        if 'DataFrame' in str(type(data)):
            D = data.copy()
        elif 'RamanChada' in str(type(data)):
            D = spectrum_to_frame(data)
        else:   
            print("Input must be DataFrame or spectrum.")
            return
        # if model is fitted
        if hasattr(self, 'x'):
            # if not x equal
            x = np.array(D.columns)
            if not np.array_equal(x, self.x):
                # make spectrum from x axis
                x_axis = Spectrum( pd.DataFrame({'x':self.x}), 'x', 'x' )
                # interpolate x onto model
                D = process_DF(D, 'interpolate_x', x_axis)
        for method, params in self.steps.items():
            D = process_DF(D, method, *params)
        self.x = np.array(D.columns)
        return D - D.min().min()

    def set_params(self, **params):
        self.steps.update(params)

    def get_params(self, deep=False):
        return self.steps

    def __repr__(self):
        return f'{self.__class__.__name__}({self.steps})'


def get_model_comps(model, step_no=1):
    # 1st step is RamanProcessor. Get the x axis from that:
    x = model.steps[0][1].x
    # The decomposer is at step_no. Get components from here:
    C = model.steps[step_no][1].components_
    comp_spectra = []
    for component in C:
        component_spectrum = pd.DataFrame({'x': x, 'y': component})
        comp_spectra.append( Spectrum(component_spectrum, 'x', 'y') )
    return RamanGroup( comp_spectra)