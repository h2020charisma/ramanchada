# + tags=["parameters"]
upstream = None
product = None
folder_native = None
force_chada_generation = None
# -

from processors import FileProcessor

folder_chada=product["data"]
FP = FileProcessor(folder_native,folder_chada,force_chada_generation)

print(folder_native,folder_chada)
FP.process_file(folder_native,recurse=True,callback=FP.parse_raman)
print("Done")

