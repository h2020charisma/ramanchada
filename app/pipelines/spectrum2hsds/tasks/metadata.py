# + tags=["parameters"]
upstream = None
product = None
root_folder = None
config_input = None
# -

from re import L
import pandas as pd
import os,path
import json

def read_opticalpath(sheet_name,file_metadata):
    
    tag_sample=["S0N","S0B","S0P","S1N","nCAL","sCAL","PST","Sil"]
    op_sheet = pd.read_excel(file_metadata,sheet_name=sheet_name,header=None)
    start_row = 29
    offset_row = 33
    tmp = []
    try:
        while not pd.isna(op_sheet.iloc[start_row][0]):
            
            for sample in range(0,len(tag_sample),1):
                file_name = op_sheet.iloc[start_row+sample][11]
                if not pd.isna(file_name):
                    files = file_name.strip().replace("\n",",").split(",")
                    files = [s.strip() for s in files]
                    tmp.append({"sample" : tag_sample[sample],"file" : files , "laser_power" : op_sheet.iloc[start_row][4]})
            start_row = start_row + offset_row
            if start_row>op_sheet.shape[0]:
                break
    except Exception as err:
        print(err,sheet_name)
        pass
    
    return tmp
    
def get(folder_input,file_metadata,product_metadata):

    front_sheet = pd.read_excel(file_metadata,sheet_name="Front sheet",header=None)
    front_sheet.fillna("",inplace=True)
    
    metadata={  "folder_input" : folder_input,
                "file_metadata" : file_metadata,
                "provider" : front_sheet.iloc[0][1], 
               "instrument" : front_sheet.iloc[0][6], "wavelength" : front_sheet.iloc[1][6],
               "optical_path" : [] }
    
    ops = metadata["optical_path"]

    row = 6
    #while not pd.isna(front_sheet.iloc[row][0]):
    try:
        while (front_sheet.iloc[row][0]!=""):
            op_id = front_sheet.iloc[row][0]
            op = {"id" : op_id,
                    "collection_optics" : front_sheet.iloc[row][2],
                    "slit_size" : front_sheet.iloc[row][4],
                    "gratings" : front_sheet.iloc[row][6],
                    "pin_hole_size" : front_sheet.iloc[row][8],
                    "collection_fibre_diameter" : front_sheet.iloc[row][10],
                    "notes" : front_sheet.iloc[row][12]}
            #print(op)
            op["files"] = read_opticalpath(op_id,file_metadata)
            #print(op)
            ops.append(op)
            row=row+1
    except Exception as err:
        pass
        
    with open(product_metadata, "w",encoding="utf-8") as write_file:
        json.dump(metadata, write_file, sort_keys=True, indent=4)    


def read_metadata(root_folder,config_input,product):
    if not os.path.exists(product["data"]):
        os.mkdir(product["data"])
    with open(config_input, 'r') as infile:
        config = json.load(infile)
    for entry in config:
        if entry["enabled"]:
            product_metadata = os.path.join(product["data"],"metadata_{}_{}_{}.json".
                format(entry["hsds_provider"],entry["hsds_instrument"],entry["hsds_wavelength"]))

            ##_root = os.path.join(root_folder,entry["folder_input"]) #not sure why it doesn't work and strips the win part to d:\
            folder_input = os.path.join(root_folder,entry["folder_input"])
            file_metadata = os.path.join(root_folder,entry["file_metadata"])
            get(folder_input,file_metadata,product_metadata)