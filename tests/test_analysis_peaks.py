from ramanchada.analysis import peaks
import numpy as np


# create one Gaussian peak with position, amplitude, width, and a noise floor
def onepeak(x, position, amplitude, width, noise):
    return (
        amplitude * np.exp(-np.square((x - position) / width))
        + amplitude * noise * np.random.normal()
    )


# test of peak finding with one Gaussian peak
# peak finding should be independent of the amplitude
def test_find_spectrum_peaks_cwt_01():
    signallength = 100
    position = 50
    amplitude = 1024
    width = 10
    noise = 0.01
    np.random.seed(17)
    x = np.array(list(x for x in range(0, signallength)))
    for amplitude in [1 / 1024, 1, 100, 1024, ]:
        y = np.array(
            list(
                onepeak(x, position, amplitude, width, noise)
                for x in range(0, signallength)
            )
        )
        P = peaks.find_spectrum_peaks_cwt(x, y)
        assert P.shape[0] >= 1
        assert np.isclose(P["intensity"][0], amplitude, 5e-2, 0)
        assert np.isclose(P["position"][0], position, 0, 1)


# test of peak finding with one Gaussian peak
# peak finding should be independent of the position
def test_find_spectrum_peaks_cwt_02():
    signallength = 100
    amplitude = 100
    width = 10
    noise = 0.01
    np.random.seed(17)
    x = np.array(list(x for x in range(0, signallength)))
    for position in [30, 50, 70, ]:
        y = np.array(
            list(
                onepeak(x, position, amplitude, width, noise)
                for x in range(0, signallength)
            )
        )
        P = peaks.find_spectrum_peaks_cwt(x, y)
        assert P.shape[0] >= 1
        assert np.isclose(P["intensity"][0], amplitude, 5e-2, 0)
        assert np.isclose(P["position"][0], position, 0, 1)


# create two Gaussian peak with position, amplitude, width, and a noise floor
def twopeaks(x, position1, amplitude1, width1, position2, amplitude2, width2, noise):
    return (
        amplitude1 * np.exp(-np.square((x - position1) / width1))
        + amplitude2 * np.exp(-np.square((x - position2) / width2))
        + np.fmax(amplitude1, amplitude2) * noise * np.random.normal()
    )


# test of peak finding with two Gaussian peaks well separated from each other
# peak finding should be independent of the amplitude
def test_find_spectrum_peaks_cwt_03():
    signallength = 300
    position1 = 100
    position2 = 200
    width = 10
    noise = 0.01
    np.random.seed(17)
    x = np.array(list(x for x in range(0, signallength)))
    for amplitude in [1 / 1024, 1, 100, 1024, ]:
        y = np.array(
            list(
                twopeaks(x, position1, amplitude, width, position2, amplitude / 2, width, noise)
                for x in range(0, signallength)
            )
        )
        P = peaks.find_spectrum_peaks_cwt(x, y)
        assert P.shape[0] >= 2
        assert np.isclose(P["intensity"][0], amplitude, 5e-2, 0)
        assert np.isclose(P["position"][0], position1, 0, 1)
        assert np.isclose(P["intensity"][1], amplitude / 2, 5e-2, 0)
        assert np.isclose(P["position"][1], position2, 0, 1)


# test of peak finding with two Gaussian peaks well separated from each other
# peak finding should be independent of the positions and its order
def test_find_spectrum_peaks_cwt_04():
    signallength = 300
    amplitude = 1024
    width = 10
    noise = 0.01
    np.random.seed(17)
    x = np.array(list(x for x in range(0, signallength)))
    for (position1, position2) in [(100, 200), (200, 100), (50, 150), (150, 50), (150, 250), (250, 150), ]:
        y = np.array(
            list(
                twopeaks(x, position1, amplitude, width, position2, amplitude / 2, width, noise)
                for x in range(0, signallength)
            )
        )
        P = peaks.find_spectrum_peaks_cwt(x, y)
        assert P.shape[0] >= 2
        assert np.isclose(P["intensity"][0], amplitude, 5e-2, 0)
        assert np.isclose(P["position"][0], position1, 0, 1)
        assert np.isclose(P["intensity"][1], amplitude / 2, 5e-2, 0)
        assert np.isclose(P["position"][1], position2, 0, 1)


# test of peak finding with two Gaussian peaks close together
# peak finding should be independent of the positions and its order
def test_find_spectrum_peaks_cwt_05():
    signallength = 300
    amplitude = 1024
    width = 10
    noise = 0.01
    np.random.seed(17)
    x = np.array(list(x for x in range(0, signallength)))
    for (position1, position2) in [(100, 125), (125, 100), (200, 225), (225, 200), ]:
        y = np.array(
            list(
                twopeaks(x, position1, amplitude, width, position2, amplitude / 2, width, noise)
                for x in range(0, signallength)
            )
        )
        P = peaks.find_spectrum_peaks_cwt(x, y)
        assert P.shape[0] >= 2
        # the positions are now not as accurate as for one peak
        assert np.isclose(P["position"][0], position1, 0, 3)
        assert np.isclose(P["position"][1], position2, 0, 4)
        assert np.isclose(P["intensity"][0], y[position1], 5e-2, 0)
        # 2nd peak amplitude deviates by 15 percent
        assert np.isclose(P["intensity"][1], y[position2], 15e-2, 0)
