from logging import ERROR
import h5pyd, h5py
import h5pyd._apps.hsinfo as hsinfo
import h5pyd._apps.hsload as hsload
import h5pyd._apps.utillib as utillib
import os, tempfile, shutil
from ramanchada.classes import RamanChada
from ramanchada.file_io.io import create_chada_from_native

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



def check_folder(self,domain="/Round_Robin_1/",create=False):
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

def load_native(file,f_name,destination_domain):
    native_filename=None
    try:
        filename, file_extension = os.path.splitext(f_name)
        #all this is because ramanchada works fith file paths only, no url nor file objects
        with tempfile.NamedTemporaryFile(delete=False,prefix="charisma_",suffix=file_extension) as tmp:
            shutil.copyfileobj(file,tmp)
            native_filename = tmp.name
        
        chada_filename = native_filename.replace(file_extension,".cha")

        create_chada_from_native(native_filename, chada_filename)
        with h5py.File(chada_filename,'r') as f_in:
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