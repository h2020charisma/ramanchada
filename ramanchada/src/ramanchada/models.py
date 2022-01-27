from copy import deepcopy
import numpy as np
import pandas as pd

from .classes import process_DF, RamanGroup, Spectrum, spectrum_to_frame

class RamanProcessor():
    """
    A class representing a transformer for Scikit-Learn. Transforms RamanGroup data in terms of Raman pre-processing steps.
    """
    def __init__(self, **steps):
        """
        Parameters
        ----------
        steps : dict
            > dict with the names of RamanChada methods as keys and the corresponding parameters as values.
            Refer to the *ramanchada.classes* documentation for details.
            Example:

                steps = {
                'x_crop': [200, 1500],
                'smooth': ['sg', 17, 2],
                'remove_baseline': [],
                'normalize': ['snv']
                }

        The default is `{'fit_baseline': ['snip'], 'remove_baseline': [], 'normalize': ['snv']}`.

        Returns
        -------
        None.

        """
        if steps == {}:
            steps = {'fit_baseline': ['snip'], 'remove_baseline': [], 'normalize': ['snv']}
        self.steps = steps

    def fit(self, df, params=None):
        """
        Parameters
        ----------
        df : DataFrame
            > Data on which the transformer is applied.
            Needed to define the .x attribute of Raman shifts.

        Returns
        -------
        None.

        """
        self.x = np.array(df.columns)
        return self

    def transform(self, data):
        """
        Parameters
        ----------
        data : DataFrame or RamanChada
            > Data to transform.
            *DataFrame* is the format for model training (*RamanGroup*.data),
            while *RamanChada* is for prediction based on a single spectrum.

        Returns
        -------
        DataFrame
            > Transformed Raman data (single or multi-row).
        """
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
        """
        Updates the steps (methods and parameters) included in the *RamanProcessor*.
        Parameters
        ----------
        params : dict
            > dict with the names of RamanChada methods as keys and the corresponding parameters as values.
            Refer to the *ramanchada.classes* documentation for details.
        
        Returns
        -------
        DataFrame
            > Transformed Raman data (single or multi-row).
        """
        self.steps.update(params)

    def get_params(self, deep=False):
        """
        Return the steps of a *RamanProcessor* as dict.
        Parameters
        ----------
        None.
        
        Returns
        -------
        dict
            > steps of the *RamanProcessor*.
        """
        return self.steps

    def __repr__(self):
        return f'{self.__class__.__name__}({self.steps})'


def get_model_comps(model, step_no=1):
    """
    Return the components of a decomposer included in a *Pipeline*,
    such as NMF or PCA, as a RamanGroup.
    Parameters
    ----------
    step_no : int
        > Index of the decomposer in the *Pipeline*.
    
    Returns
    -------
    RamanGroup
        > Components as Raman spectra in a *RamanGroup*.
    """
    # 1st step is RamanProcessor. Get the x axis from that:
    x = model.steps[0][1].x
    # The decomposer is at step_no. Get components from here:
    C = model.steps[step_no][1].components_
    comp_spectra = []
    for component in C:
        component_spectrum = pd.DataFrame({'x': x, 'y': component})
        comp_spectra.append( Spectrum(component_spectrum, 'x', 'y') )
    return RamanGroup( comp_spectra)