# + tags=["parameters"]
upstream = None
product = None
folder_native = None
folder_chada = None
# -

from FileProcessor import FileProcessor

FP = FileProcessor(folder_native,folder_chada)

print(folder_native,folder_chada)
FP.process_file(folder_native,recurse=True,callback=FP.parse_raman)
print("Done")
