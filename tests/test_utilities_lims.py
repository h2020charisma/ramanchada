from ramanchada.utilities import lims
import numpy as np
import sys


# Testing lims with xmin <= smallest x value of the array and
# xmax >= largest x value of the array should always
# return the entire array
def test_lims_01():
    x = np.arange(0, 4000, dtype="float")
    for xmin, xmax in [
        (0, 4000),
        (-10000, 10000),
        (-1e9, 1e9),
        (sys.float_info.min, sys.float_info.max),
        (-float("inf"), float("inf")),
    ]:
        lf = lims(x, xmin, xmax)
        xx = lf(x)
        assert len(xx) == len(x)
        assert xx[0] == x[0]
        assert xx[-1] == x[-1]


# Testing lims with xmin inbetween smallest and largest value of the array
# should result in an array that starts with xmin
def test_lims_02():
    x = np.arange(0, 4000, dtype="float")
    for xmax in [4000, 4001, 1e9, sys.float_info.max, float("inf")]:
        for xmin in [
            0,
            1,
            17,
            39,
            3333,
            3999,
        ]:
            lf = lims(x, xmin, xmax)
            xx = lf(x)
            assert len(xx) <= len(x)
            assert xx[0] == xmin
            assert xx[-1] == x[-1]


# Testing lims with xmax inbetween smallest and largest value of the array
# should result in an array that ends with xmax
def test_lims_03():
    x = np.arange(0, 4000, dtype="float")
    for xmin in [
        0,
        -1,
        -1e9,
        sys.float_info.min,
        -float("inf"),
    ]:
        for xmax in [
            1,
            17,
            39,
            3333,
            3999,
        ]:
            lf = lims(x, xmin, xmax)
            xx = lf(x)
            assert len(xx) <= len(x)
            assert xx[0] == x[0]
            assert xx[-1] == xmax
