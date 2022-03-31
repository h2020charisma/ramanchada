from ramanchada.analysis import signal
import numpy as np

# signal to noise ratio of a 0,1,0,1,.. sequence should be 0.5
# and should be independent on the sequence length
def test_snr_01():
    for nitems in [10, 100, 1000]:
        y = np.array(list(x%2 for x in range(0,nitems)))
        snr = signal.snr(y)
        assert snr == 0.5



# signal to noise ratio should be independent of an offset,
# thus adding an offset to the sequence 0,1,0,1,.. 
# should still result in snr of 0.5
def test_snr_02():
    for offset in [-33, 1, 17]:
        y = np.array(list(offset+x%2 for x in range(0,10)))
        snr = signal.snr(y)
        assert snr == 0.5


# signal to noise ratio should be independent of the y-scale,
# thus multiplying a scale to the sequence 0,1,0,1,.. 
# should still result in snr of 0.5
def test_snr_03():
    for scale in [-7, 3, 13]:
        y = np.array(list(17+scale*(x%2) for x in range(0,10)))
        snr = signal.snr(y)
        assert snr == 0.5
