from logging import ERROR
import h5pyd, h5py
import h5pyd._apps.hsinfo as hsinfo
import h5pyd._apps.hsload as hsload
import h5pyd._apps.utillib as utillib
import os, tempfile, shutil
from ramanchada.classes import RamanChada
from ramanchada.file_io.io import create_chada_from_native
from  tempfile import TemporaryFile

from enum import Enum
import time
import uuid
import json



class TaskStatus(Enum):
    Running="Running", 
    Cancelled="Cancelled",
    Completed="Completed",
    Error="Error",
    Queued="Queued"

    

class TaskResult:
    def __init__(self,name=None,status : TaskStatus=None,started=time.ctime()):
        self.uri = None
        self.id = str(uuid.uuid4())
        self.name = name
        self.error = None
        self.status = str(status)
        self.started=started
        self.completed = None
        self.result = None
        self.errorCause=None
    
    def set_completed(self,result,completed=time.ctime()):
        self.status = str(TaskStatus.Completed)
        self.result = result
        self.completed = completed

    def set_queued(self,result,completed=time.ctime()):
        self.status = str(TaskStatus.Queued)
        self.result = result
        self.completed = completed
     
    def set_cancelled(self,error,errorCause,completed=time.time()):
        self.status = str(TaskStatus.Cancelled)
        self.error=error
        self.errorCause=errorCause
        self.result = None
        self.completed = completed     

    def set_error(self,error,errorCause=None,completed=time.ctime()):
        self.status = str(TaskStatus.Error)
        self.error=error
        self.errorCause=errorCause
        self.result = None
        self.completed = completed    

    def to_dict(self):
        return {"name" : self.name,
            "result" : self.result, "error" : self.error, "errorCause" : self.errorCause,
             "completed" : self.completed, "started" : self.started, "status" : self.status }  


def read_file(domain=None):
    cfg = hsinfo.cfg
    try:
        return h5pyd.File(domain, endpoint=cfg["hs_endpoint"], username=cfg["hs_username"], password=cfg["hs_password"], bucket="charisma", owner=cfg["hs_username"])
    except IOError as err:
        raise err

def get_file_annotations(file=None):
    annotation = {}
    datasets = []
    for key in file.keys():
        print(key)
        if key=="annotation_sample":
            for item in file[key].attrs:
                annotation[item]=file[key].attrs[item] 
        elif key=="annotation_study":
            for item in file[key].attrs:
                annotation[item]=file[key].attrs[item]  
        else:
            datasets.append({"key" : key, "uuid" : file[key].id.uuid, "name" : file[key].name, 
                                "shape" : file[key].shape, "size" : file[key].size})   
    print(annotation)
    print(datasets)
    return annotation,datasets

def check_folder(domain="/Round_Robin_1/",create=False):
    cfg = hsinfo.cfg
    try:
        return h5pyd.Folder(domain, endpoint=cfg["hs_endpoint"], username=cfg["hs_username"], password=cfg["hs_password"], bucket="charisma", owner=cfg["hs_username"])
    except IOError as err:
        if create:
            return h5pyd.Folder(domain, mode='x',  endpoint=cfg["hs_endpoint"], username=cfg["hs_username"], password=cfg["hs_password"], bucket="charisma", owner=cfg["hs_username"])
        else:
            raise err


def load_h5stream(stream,destination_domain):
    try:
        f_in = h5py.File(stream,'r')
        load_h5file(f_in,destination_domain)
    except Exception as err:
        raise

def load_h5file(h5file,destination_domain):
    try:
        cfg = hsinfo.cfg
        with h5pyd.File(destination_domain, "w", endpoint=cfg["hs_endpoint"], username=cfg["hs_username"], password=cfg["hs_password"], bucket="charisma", retries=3, owner=cfg["hs_username"]) as fout:
            utillib.load_file(h5file, fout, verbose=False,dataload="ingest")    
    except Exception as err:
        raise    

def cha2tmp(fname,params={}):
    tmpfile = TemporaryFile()
    tf = h5py.File(tmpfile, 'w')
    #attributes = self.parse_name(fname)
    try:
        with h5py.File(fname, "r") as f:       
            _group = tf.create_group("annotation_study")  
            for p in params:
                if p!="sample":
                    try:
                        _group.attrs.create(p,params[p])
                    except Exception as err:
                        print(err)   
            
            try:
                _group = tf.create_group("annotation_sample")  
                p="sample"
                _group.attrs.create(p,params[p])
            except Exception as err:
                print(err)                       

            for key in f.keys():
                _tmp=f[key]
                _dataset = tf.create_dataset(key, data=_tmp)
                for item in f[key].attrs.items():
                    _dataset.attrs.create(item[0],item[1])
        
    except Exception as err:
        print(">>",err)

    return tf 

def load_native(file,f_name,destination_domain,params={}):
    native_filename=None
    try:
        filename, file_extension = os.path.splitext(f_name)
        #all this is because ramanchada works fith file paths only, no url nor file objects
        with tempfile.NamedTemporaryFile(delete=False,prefix="charisma_",suffix=file_extension) as tmp:
            shutil.copyfileobj(file,tmp)
            native_filename = tmp.name
        
        chada_filename = native_filename.replace(file_extension,".cha")

        create_chada_from_native(native_filename, chada_filename)
        try:
            R = RamanChada(chada_filename)
            R.normalize()
            R.fit_baseline()
            R.remove_baseline()
            R.commit("baseline_removed")
            params["native_filename"] = os.path.basename(f_name)
        except Exception as err:
            print(chada_filename,err)
        
        with cha2tmp(chada_filename,params) as f_in: 
            load_h5file(f_in,destination_domain)
    except Exception as err:
        print(err)
        raise          
    finally:
        if native_filename!=None:
            os.remove(native_filename)
            
        if chada_filename!=None:
            try:
                os.remove(chada_filename) 
            except:
                pass           
  
from ramanchada.classes import RamanChada
def load_domain(url,raw=True):
    print(url)
    f = None
    try:
        #url="/Round_Robin_1/LBF/nCAL10_iR532_Probe_005_2500msx3.cha"
        R = RamanChada(url,raw=raw,is_h5pyd=True)
        #f = h5pyd.File(url, 'r')
        
        return R

    except Exception as err:
        raise err
    finally:
        if f!=None:
            f.close()