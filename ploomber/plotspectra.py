# + tags=["parameters"]
upstream = None
product = None
upstream = ['normalise']
folder_chada = None
# -


import os
from ramanchada.classes import RamanChada
from ramanchada.file_io.io import get_chada_commits

def plot_spectra(f_name):
    filename, file_extension = os.path.splitext(f_name)
    if file_extension ==".cha":       
        commits = get_chada_commits(f_name)        
        print(f_name,commits)     
        SOP = RamanChada(f_name,raw=True)
        SOP.plot()           
        SOP = RamanChada(f_name,raw=False)
        SOP.plot()

from processors import FileProcessor

FP = FileProcessor(folder_chada,folder_chada)

FP.process_file(folder_chada,recurse=True,callback=plot_spectra)
print("Done")