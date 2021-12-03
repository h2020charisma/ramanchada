import os,path
from ramanchada.file_io.io import create_chada_from_native
import logging,sys,os

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class FileProcessor:
    def __init__(self,folder_input,folder_output,force_chada_generation=False):
        self.folder_input=folder_input
        self.folder_output=folder_output
        self.force_chada_generation = force_chada_generation

    def parse_raman(self,native_filename):
        filename, file_extension = os.path.splitext(native_filename)
        print(filename)
        if file_extension==".xlsx":
            return None
        if file_extension!=".cha":     
            print("Parsing file: '%s' ..." % native_filename, end="\n")
            try:
                chada_filename = os.path.join(self.folder_output,os.path.basename(filename)+".cha")
                ##self.walkpath(chada_filename,callback=lambda head,tail: print(">>",head,">>",tail))
                if self.force_chada_generation or (not os.path.exists(chada_filename)):
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

    def walkpath(self,path = None,callback=None):
        if path is None or path=="" or  path.endswith("/"):
            return None
        head, tail = os.path.split(path)

        self.walkpath(head,callback)  
        if callback!=None:
            callback(head,tail)                  

import h5pyd
import h5py
import h5pyd._apps.hsload as hsload
import h5pyd._apps.hsinfo as hsinfo
import h5pyd._apps.utillib as utillib
import h5pyd._apps.hsdel as hsdel
import uuid, h5py
from  tempfile import TemporaryFile

class Teleport:
    def __init__(self,cfg,bucket="charisma",owner=None,domain="/",brands=None):
        self.cfg=cfg
        self.bucket=bucket
        self.set_owner(owner)
        self.set_brands(brands)
        self.h5object= None
    

    def set_brands(self,brands):
        self.brands = brands
    
    def set_domain(self,domain="/RoundRobin1/"):
        self.domain=domain

    def set_owner(self,owner=None):
        if owner is None:
            self.owner=self.cfg["hs_username"]
        else:
            self.owner=owner

    def check_folder(self,domain="/Round_Robin_1/"):
        try:
            dir = h5pyd.Folder(domain, endpoint=self.cfg["hs_endpoint"], username=self.cfg["hs_username"], password=self.cfg["hs_password"], bucket=self.bucket, owner=self.owner)
        except IOError as err:
            dir = h5pyd.Folder(domain, mode='x',  endpoint=self.cfg["hs_endpoint"], username=self.cfg["hs_username"], password=self.cfg["hs_password"], bucket=self.bucket, owner=self.owner)
        return dir

    def delete_domain_recursive(self,fname):
        try:
            #print(fname)
            f = h5pyd.Folder(fname+"/")
            n = f._getSubdomains()
            if n>0:
                for s in f._subdomains:
                    #print(s["name"])
                    self.delete_domain_recursive(s["name"])
        except Exception as err:
            print(">>>",fname,err)
        try:
            #print("Deleting ",fname)    
            hsdel.deleteDomain(fname)
        except Exception as err:
            print(err)

#[Sample name]_[instrument & laser wavelength]_[optical path]_[power]_[timexavgs] 
    def parse_name(self,f_name):
        tags = os.path.basename(f_name).replace(".cha","").split("_") 
        timexavgs = tags[4].split("x")
        print(len(tags),tags)
        misc=None
        if len(tags)>5:
            misc = "_".join(tags[5:])
        instrument = tags[1][0:-3].upper();    
        instrument_brand=""
        instrument_model=""   
        for b in self.brands:
            if instrument.startswith(b):
                instrument_brand = b
                instrument_model = instrument.replace(b,"")
                break;  
        results = {
            "sample" : tags[0],
            "instrument" : instrument,
            "instrument_brand" : instrument_brand,
            "instrument_model" : instrument_model,
            "wavelength" : tags[1][-3:],
            "optical_path" : tags[2],
            "power" : tags[3],
            "time"  : timexavgs[0],
            "scans" : timexavgs[1]
        }
        if misc != None:
            results["misc"] = misc
        return results

    # add chada files into same h5py temporary file
    def cha2study(self,fname):
        filename, file_extension = os.path.splitext(fname)
        if file_extension!=".cha": 
            return None
        if self.h5object is None:
            tmpfile = TemporaryFile()
            self.h5object = h5py.File(tmpfile, 'w')
            _group = self.h5object.create_group("investigation")
        else:
            _group = self.h5object["investigation"]


        try:
            attributes = self.parse_name(fname)
            dest = os.path.basename(fname)
            _subgroup = _group.create_group(dest)            
            with h5py.File(fname, "r") as f:       
                for item in attributes.items():
                    try:
                        _subgroup.attrs.create(item[0],item[1])
                    except Exception as err:
                        print(item,err)   

                for key in f.keys():
                    _tmp=f[key]
                    _dataset = _subgroup.create_dataset(key, data=_tmp)
                    for item in f[key].attrs.items():
                        _dataset.attrs.create(item[0],item[1])
        except Exception as err:
            print(">>",err)

        return self.h5object 
        
    def cha2tmp(self,fname):
        tmpfile = TemporaryFile()
        tf = h5py.File(tmpfile, 'w')
        attributes = self.parse_name(fname)
        try:
            with h5py.File(fname, "r") as f:       
                _group = tf.create_group("root")  
                for item in attributes.items():
                    try:
                        _group.attrs.create(item[0],item[1])
                    except Exception as err:
                        print(item,err)   

                for key in f.keys():
                    _tmp=f[key]
                    _dataset = _group.create_dataset(key, data=_tmp)
                    for item in f[key].attrs.items():
                        _dataset.attrs.create(item[0],item[1])
        except Exception as err:
            print(">>",err)

        return tf 

    def study2hsds(self,destination):
        if self.h5object is None:
            return None
        try:
            fout = h5pyd.File(destination, "w", endpoint=self.cfg["hs_endpoint"], username=self.cfg["hs_username"], password=self.cfg["hs_password"], bucket=self.bucket, retries=3, owner=self.owner)
            utillib.load_file(self.h5object, fout, verbose=False,dataload="ingest")
            
        except Exception as err:
                print(err)     
              
              

    def chada2hsds(self,fname,delete=False):
        filename, file_extension = os.path.splitext(fname)
        dest=None
        if file_extension==".cha":     
            
            try:
                self.check_folder(self.domain);           
            except Exception as err:
                print(self.domain,self.owner,err)
            #print("Loading file: '%s' ..." % fname, end="\n")     

            fin=None
            fout=None       
            try:
                dest = "{}{}".format(self.domain,os.path.basename(fname))
                #.replace(".cha",""))
                
                if delete:
                    try:
                        print("Delete ",dest)
                        hsdel.deleteDomain(dest)
                    except Exception as err:
                        print(err)                      

                fin = self.cha2tmp(fname)
                fout = h5pyd.File(dest, "w", endpoint=self.cfg["hs_endpoint"], username=self.cfg["hs_username"], password=self.cfg["hs_password"], bucket=self.bucket, retries=3, owner=self.owner)
                utillib.load_file(fin, fout, verbose=False,dataload="ingest")
                return dest
            except Exception as err:
                print(err)
               

#[Sample name]_[instrument & laser wavelength]_[optical path]_[power]_[timexavgs] 
def parse_name(f_name,brands=[]):
    tags = os.path.basename(f_name).replace(".cha","").split("_") 
    print(f_name,len(tags),tags)
    if len(tags)>4:
        timexavgs = tags[4].split("x")
    else:
        timexavgs=[None,None]
    misc=None
    if len(tags)>5:
        misc = "_".join(tags[5:])
    instrument = tags[1][0:-3].upper();    
    instrument_brand=""
    instrument_model=""   
    for b in brands:
        if instrument.startswith(b):
            instrument_brand = b
            instrument_model = instrument.replace(b,"")
        break;  
    results = {
        "sample" : tags[0],
        "instrument" : instrument,
        "instrument_brand" : instrument_brand,
        "instrument_model" : instrument_model,
        "wavelength" : tags[1][-3:],
        "optical_path" : tags[2],
        "power" : tags[3],
        "time"  : timexavgs[0],
        "scans" : timexavgs[1]
    }
    if misc != None:
        results["misc"] = misc
    return results               