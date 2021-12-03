import os,path
from ramanchada.file_io.io import create_chada_from_native

class FileProcessor:
    def __init__(self,folder_input,folder_output):
        self.folder_input=folder_input
        self.folder_output=folder_output

    def parse_raman(self,native_filename):
        filename, file_extension = os.path.splitext(native_filename)
        if file_extension==".xlsx":
            return None
        if file_extension!=".cha":     
            print("Parsing file: '%s' ..." % native_filename, end="\n")
            try:
                chada_filename = os.path.join(self.folder_output,os.path.basename(filename)+".cha")
                if not os.path.exists(chada_filename):
                    create_chada_from_native(native_filename,chada_filename)
                
            except Exception as err:
                print(err)


    def process_file(self,f_name, recurse=False,callback=parse_raman):
        if os.path.isfile(f_name):
            return [callback(f_name)]
        elif os.path.isdir(f_name) and recurse:
            return [callback(os.path.join(root, f)) for (root, dirs, files) in os.walk(f_name) for f in files]
        else:
            return []