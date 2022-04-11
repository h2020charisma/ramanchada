from pathlib import Path
from ramanchada.classes import RamanChada
import os


def get_native_file(filename='200218-17.wdf'):
    TEST_FILE_DIR = Path('test')
    TEST_FILE = TEST_FILE_DIR / filename
    assert TEST_FILE.exists()
    return TEST_FILE


def get_chada_file(filename='200218-17.cha'):
    TEST_FILE_DIR = Path('test')
    TEST_FILE = TEST_FILE_DIR / filename
    return TEST_FILE


# This tests if RamanChada can load a file.
def test_load_data_file_into_object():
    native_file = get_native_file()
    chada_obj = RamanChada(native_file)
    assert chada_obj.x.shape == (3190,)

    chada_obj.fit_baseline(method='snip')
    chada_obj.remove_baseline()

    chada_obj.rewind(0)
    chada_obj.fit_baseline(method='als')
    chada_obj.remove_baseline()

    chada_obj.normalize("area")
    chada_obj.peaks(fitmethod='dvoigt')
    print(chada_obj.bands)

    chada_obj.peaks(fitmethod='gl')
    chada_obj.peaks(fitmethod='voigt')
    # Clean up.
    chada_file = get_chada_file()
    if chada_file.exists():
        os.remove(chada_file)

def test_file_open_close():
    native_file = get_native_file("Polystyrene.spc")
    co1 = RamanChada(native_file)
    
    co2 = RamanChada(native_file)   

#def test_file_l6s():
#    native_file = get_native_file("PST10_iR785_OP02_8000msx8_01.l6s")
#    co1 = RamanChada(native_file)
    