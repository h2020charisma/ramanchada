# + tags=["parameters"]
upstream = None
product = None
upstream = ['native2chada']
# -


import os
from ramanchada.classes import RamanChada

import sys
import logging



def normalise(f_name):
    filename, file_extension = os.path.splitext(f_name)
    if file_extension ==".cha":
        logger.info(f_name)
        R = RamanChada(f_name,raw=True)
        R.normalize('minmax')
        R.commit("normalized_minmax")
        R.fit_baseline()
        R.remove_baseline()
        R.commit("baseline_removed")

from processors import FileProcessor

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

folder_chada=upstream['native2chada']['data']
FP = FileProcessor(folder_chada,folder_chada)

FP.process_file(folder_chada,recurse=True,callback=normalise)
print("Done")