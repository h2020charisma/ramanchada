# + tags=["parameters"]
upstream = ["tasks.metadata.read_metadata"]
product = None
config_input = None
metadata_root = None
ramandb_api = None
hs_username = None
hs_password = None
hsds_investigation = None

# -

import json

import os,path
import requests
from pprint import pprint
import glob 

def submit2hsds(file_name,ramandb_api,hs_username,hs_password,
                hsds_investigation,hsds_provider,hsds_instrument,hsds_wavelength,optical_path,sample,laser_power,metadata):
    domain = "/{}/{}/{}/{}/".format(hsds_investigation,hsds_provider,hsds_instrument,hsds_wavelength)
    api_dataset = "{}dataset?domain={}".format(ramandb_api,domain);
    formData = {"investigation" :hsds_investigation,
	            "provider":hsds_provider,
	            "instrument": hsds_instrument,
		        "wavelength": hsds_wavelength,
                "optical_path" : optical_path,
                "sample" : sample,
                "laser_power" : laser_power}
    with  open(file_name,'rb') as _file:
        formFiles = {"file[]" :  _file}
            #formData.append("optical_path",$("#optical_path").val());
            #formData.append("laser_power",$("#laser_power").val());
        response = requests.post(api_dataset, data=formData,files=formFiles,auth=(hs_username,hs_password))
        metadata[file_name] = response.json()
    
import pandas as pd

def files2hsds(metadata,ramandb_api,hs_username,hs_password,hsds_investigation,hsds_provider,hsds_instrument,hsds_wavelength,log_file):
    folder_input = metadata["folder_input"]
    log = {"results" : {}}
    spectra_files = glob.glob(os.path.join(folder_input,"**","*"), recursive=True)
    file_lookup = {}
    for file_name in spectra_files:
        if file_name.endswith(".xlsx"):
            continue
        if os.path.isdir(file_name):
            continue
        basename = os.path.basename(file_name)
        (base,ext) = os.path.splitext(basename)
        file_lookup[basename] = [file_name]
        if base in file_lookup:
            file_lookup[base].append(file_name)
        else:
            file_lookup[base]  = [file_name]

    for op in metadata["optical_path"]:
        for files in op["files"]:
            for file in files["file"]:
                if file in file_lookup:
                    for file_name in file_lookup[file]:
                        laser_power = "" if pd.isna(files["laser_power"]) else files["laser_power"]
                        sample = files["sample"] 
                        op_id = op["id"]
                        print(sample,op_id,laser_power,file_name)
                        submit2hsds(file_name,ramandb_api,hs_username,hs_password,
                            hsds_investigation,hsds_provider,hsds_instrument,hsds_wavelength,
                            op_id,sample,laser_power,log["results"])
            

    log["files"] = file_lookup
    with open(log_file, "w",encoding="utf-8") as write_file:
        json.dump(log, write_file, sort_keys=True, indent=4)    

def folders2hsds(config_input,metadata_root,ramandb_api,hs_username,hs_password,hsds_investigation,product):
    
    with open(config_input, 'r') as infile:
        config = json.load(infile)
    for entry in config:
        if entry["enabled"]:
            json_metadata = os.path.join(metadata_root,"metadata_{}_{}_{}.json".
                format(entry["hsds_provider"],entry["hsds_instrument"],entry["hsds_wavelength"]))
            log_file = os.path.join(product["data"],"log_{}_{}_{}.json".
                format(entry["hsds_provider"],entry["hsds_instrument"],entry["hsds_wavelength"]))
            with open(json_metadata, 'r') as infile:
                metadata = json.load(infile)

            files2hsds(metadata,ramandb_api,hs_username,hs_password,
                    hsds_investigation,entry["hsds_provider"],entry["hsds_instrument"],
                    entry["hsds_wavelength"],log_file)
    