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


def test_lims_04():
    x = np.arange(-100, 200, dtype=float)
    y = x**3

    ll = lims(x, -np.infty, np.infty)
    lx, ly = ll(x), ll(y)
    assert len(x) == len(lx), "should not crop, since [-infty to infty]"

    boundaries = np.array([-200, -102, -101, -100, -99, -98, 0, 198, 199, 200, 201, 300])
    deltas = np.array([-np.finfo(float).eps, 0, np.finfo(float).eps]) * np.max(np.abs(x))
    boundaries = np.add.outer(boundaries, deltas).reshape(-1)
    for xmin in boundaries:
        for xmax in boundaries:
            ll = lims(x, xmin, xmax)
            lx, ly = ll(x), ll(y)
            if xmin > xmax:
                assert len(lx) == 0, "inversed boundaries should lead to zero size array"
            if xmin == xmax:
                assert len(lx) in {0, 1}, "equal boundaries should lead to empty or single-element array"
            if len(lx) > 0:
                assert xmin <= lx[0], "xmin <= lx[0]"
                assert xmax >= lx[-1], "xmax >= lx[-1]"
                assert xmin < lx[0] + max(1, np.abs(lx[0]))*np.finfo(float).eps, "xmin < lx[0] + epsilon"
                assert xmax > lx[-1] - max(1, np.abs(lx[-1]))*np.finfo(float).eps, "xmax > lx[-1] - epsilon"
                if xmin in x:
                    assert lx[0] == xmin, "lower boundary should be included"
                if xmax in x:
                    assert lx[-1] == xmax, "higher boundary should be included"
            assert len(lx) == len(ly), "lx and ly should have equal lengths"
            zero = np.sum(np.abs(np.cbrt(ly) - lx))
            assert np.float32(zero + 1) == 1, "correspondance between lx and ly elements should be preserved"
