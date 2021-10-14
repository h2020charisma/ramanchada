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
    chada_obj.peaks(fitmethod='vg')
    print(chada_obj.bands)

    # Clean up.
    chada_file = get_chada_file()
    if chada_file.exists():
        os.remove(chada_file)
