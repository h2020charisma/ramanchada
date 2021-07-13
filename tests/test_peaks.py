from pathlib import Path

from ramanchada import chada
from ramanchada import chada_io


def init_data_file():
    TEST_FILE_DIR = Path('test')
    TEST_FILE = TEST_FILE_DIR / '200218-17.wdf'
    assert TEST_FILE.exists()
    return TEST_FILE


def test_peaks():
    chada_file = chada_io.create(init_data_file())
    C = chada.Chada(chada_file)
    assert C.x_data.shape == (3190,)
    C.peaks()
    assert abs(C.bands.at[5,"position"] - 1615.631) < 1E-3
    assert abs(C.bands.at[6,"position"] - 1727.773) < 1E-3
    assert abs(C.bands.at[8,"position"] - 3083.742) < 1E-3
    assert abs(C.bands.at[5,"FWHM"] - 9.335) < 1E-3
    assert abs(C.bands.at[6,"FWHM"] - 13.084) < 1E-3
    assert abs(C.bands.at[8,"FWHM"] - 13.561) < 1E-3    

def test_peaks_with_baseline():
    chada_file = chada_io.create(init_data_file())
    C = chada.Chada(chada_file)
    assert C.x_data.shape == (3190,)
    C.fit_baseline(show=True)
    C.remove_baseline()    
    C.peaks()
    assert abs(C.bands.at[5,"position"] - 1615.631) < 1E-3
    assert abs(C.bands.at[6,"position"] - 1727.773) < 1E-3
    assert abs(C.bands.at[8,"position"] - 3083.742) < 1E-3

    assert abs(C.bands.at[5,"FWHM"] - 9.328) < 1E-3
    assert abs(C.bands.at[6,"FWHM"] - 13.057) < 1E-3
    assert abs(C.bands.at[8,"FWHM"] - 13.563) < 1E-3  
