# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.1 (Prototype), Sept 28, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# Python libraries
import numpy as np


def noise(Y):
    return np.sum(np.abs(np.diff(Y, axis=-1)), axis=-1) / np.max(Y, axis=-1)

def snr(Y):
    avg_noise = np.mean(np.abs(np.diff(Y, axis=-1)), axis=-1)
    return np.std(Y, axis=-1) / avg_noise

def signal(Y):
    signal = -noise(Y)
    signal -= signal.min()
    signal /= signal.max()
    return signal