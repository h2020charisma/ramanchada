from pathlib import Path

from ramanchada import chada
from ramanchada import chada_io


def init_data_file():
    TEST_FILE_DIR = Path('test')
    TEST_FILE = TEST_FILE_DIR / '200218-17.wdf'
    assert TEST_FILE.exists()
    return TEST_FILE


def test_load_data_file_into_object():
    chada_file = chada_io.create(init_data_file())
    chada_obj = chada.Chada(chada_file)
    assert chada_obj.x_data.shape == (3190,)
