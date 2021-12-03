# + tags=["parameters"]
upstream = None
product = None
folder_chada = None
upstream = ['native2chada']
# -


import os
import pandas as pd
from ramanchada.classes import RamanChada

def plot_spectra(f_name):
    filename, file_extension = os.path.splitext(f_name)
    if file_extension ==".cha":        
        SOP = RamanChada(f_name)
        SOP.plot()

from FileProcessor import FileProcessor

FP = FileProcessor(folder_chada,folder_chada)

FP.process_file(folder_chada,recurse=True,callback=plot_spectra)
print("Done")