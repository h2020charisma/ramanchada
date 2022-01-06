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

from enum import Enum
class ParamsRaman(Enum):
    BRAND = "brand"
    MODEL = "model"
    WAVELENGTH = "wavelength"
    GRATINGS   ="gratings"
    COLLECTION_OPTICS="collection_optics"
    SLIT_SIZE = "slit_size"
    PIN_HOLE_SIZE = "pin_hole_size"
    COLLECTION_FIBRE_DIAMETER = "collection_fibre_diameter"
    OTHER = "param_other"

class StudyRegistration:
    def __init__(self):
        self._tag_instrument = "instrument"
        self._tag_optical_paths = "optical_paths"
        self._tag_id = "id"

        pass
    def get_instrument_id(self,instrument):
        if self._tag_id in instrument:
            id = instrument[self._tag_id]
        else:
            id = "{}_{}".format(instrument[ParamsRaman.BRAND.value].upper(),instrument[ParamsRaman.MODEL.value])
        return id

    def create_metadata(self,investigation,provider,
        brand = "",model="",wavelength=-1,collection_optics = [],gratings=[],slit_size=[],pin_hole_size=[]):
        instrument = self.create_instrument(brand,model,wavelength,collection_optics,gratings,slit_size,
            pin_hole_size);
        
        instrument["id"] = self.get_instrument_id(instrument)
        print(instrument)
        return {
            "investigation" : investigation,
            "provider" : provider,
            self._tag_instrument : instrument
        }
    def create_instrument(self,brand = "",model="",wavelength=-1,collection_optics = [],gratings=[],slit_size=[],
                    pin_hole_size=[],collection_fibre_diameter=[],param_other=[]):
        return {
            
 			ParamsRaman.BRAND.value: brand,
 			ParamsRaman.MODEL.value: model,
 			ParamsRaman.WAVELENGTH.value: wavelength,
 			ParamsRaman.COLLECTION_OPTICS.value: collection_optics,
 			ParamsRaman.GRATINGS.value: gratings,
 			ParamsRaman.SLIT_SIZE.value: slit_size,
 			ParamsRaman.PIN_HOLE_SIZE.value: pin_hole_size,
 			ParamsRaman.COLLECTION_FIBRE_DIAMETER.value: collection_fibre_diameter,
             ParamsRaman.OTHER.value: param_other
        }


    def instrument2h5(self,instrument,h5_group):
        h5_group.attrs[self._tag_id] = self.get_instrument_id(instrument)
        h5_group.attrs[ParamsRaman.BRAND.value] = instrument[ParamsRaman.BRAND.value]
        h5_group.attrs[ParamsRaman.MODEL.value] = instrument[ParamsRaman.MODEL.value]
        h5_group.attrs[ParamsRaman.WAVELENGTH.value] = instrument[ParamsRaman.WAVELENGTH.value]
        h5_group.attrs[ParamsRaman.COLLECTION_OPTICS.value] = instrument[ParamsRaman.COLLECTION_OPTICS.value]
        h5_group.attrs[ParamsRaman.GRATINGS.value] = instrument[ParamsRaman.GRATINGS.value]
        h5_group.attrs[ParamsRaman.SLIT_SIZE.value] = instrument[ParamsRaman.SLIT_SIZE.value]
        h5_group.attrs[ParamsRaman.PIN_HOLE_SIZE.value] = instrument[ParamsRaman.PIN_HOLE_SIZE.value]   
                #print(instrument["optical_paths"]) 
        _g_ops = h5_group.create_group(self._tag_optical_paths)
        for op in instrument[self._tag_optical_paths]:
            _g_op = _g_ops.create_group(op[self._tag_id])
            _g_op.attrs[ParamsRaman.COLLECTION_OPTICS.value] = op[ParamsRaman.COLLECTION_OPTICS.value]
            _g_op.attrs[ParamsRaman.GRATINGS.value] = op[ParamsRaman.GRATINGS.value]
            _g_op.attrs[ParamsRaman.SLIT_SIZE.value] = op[ParamsRaman.SLIT_SIZE.value]
            _g_op.attrs[ParamsRaman.PIN_HOLE_SIZE.value] = op[ParamsRaman.PIN_HOLE_SIZE.value] 


    def metadata2h5(self,metadata,h5file): 
        h5file.attrs["investigation"] = metadata["investigation"]
        h5file.attrs["provider"] = metadata["provider"]
        try:
            _g_instrument = h5file.create_group(self._tag_instrument)
            instrument = metadata[self._tag_instrument]
            self.instrument2h5(instrument,_g_instrument)
        except Exception as err:
            print(err)

    def h52metadata(self,h5file): 
        metadata = {}
        metadata["investigation"] = h5file.attrs["investigation"]
        metadata["provider"] = h5file.attrs["provider"]
        group_instrument = h5file[self._tag_instrument]
        instrument = self.h52instrument(group_instrument)
        metadata[self._tag_instrument] = instrument
                
        return metadata

    def h52instrument(self,group_instrument):
        optical_paths = []
    
        instrument = {
            self._tag_id : group_instrument.attrs[self._tag_id],
            ParamsRaman.BRAND.value : group_instrument.attrs[ParamsRaman.BRAND.value],
            ParamsRaman.MODEL.value : group_instrument.attrs[ParamsRaman.MODEL.value],
            ParamsRaman.WAVELENGTH.value : int(group_instrument.attrs[ParamsRaman.WAVELENGTH.value]),
            ParamsRaman.COLLECTION_OPTICS.value : group_instrument.attrs[ParamsRaman.COLLECTION_OPTICS.value].tolist(), #converts to python types
            ParamsRaman.SLIT_SIZE.value :  group_instrument.attrs[ParamsRaman.SLIT_SIZE.value].tolist(),
            ParamsRaman.GRATINGS.value : group_instrument.attrs[ParamsRaman.GRATINGS.value].tolist(),
            ParamsRaman.PIN_HOLE_SIZE.value :  group_instrument.attrs[ParamsRaman.PIN_HOLE_SIZE.value].tolist(),
            self._tag_optical_paths : optical_paths
        }
                
        _g_ops= group_instrument[self._tag_optical_paths]
        for op in _g_ops.keys():
            _g_op=_g_ops[op]
            optical_path = {}
            optical_path["id"] = op
            optical_path[ParamsRaman.COLLECTION_OPTICS.value] = _g_ops[op].attrs[ParamsRaman.COLLECTION_OPTICS.value]
            optical_path[ParamsRaman.SLIT_SIZE.value] = int(_g_ops[op].attrs[ParamsRaman.SLIT_SIZE.value])
            optical_path[ParamsRaman.GRATINGS.value] = int(_g_ops[op].attrs[ParamsRaman.GRATINGS.value])
            optical_path[ParamsRaman.PIN_HOLE_SIZE.value] = int(_g_ops[op].attrs[ParamsRaman.PIN_HOLE_SIZE.value] )
            optical_paths.append(optical_path) 
        return instrument                         


                  