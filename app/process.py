import h5pyd, h5py
import h5pyd._apps.hsinfo as hsinfo
import h5pyd._apps.hsload as hsload
import h5pyd._apps.utillib as utillib
import os, tempfile, shutil
from ramanchada.file_io.io import create_chada_from_native


def load_h5stream(stream,destination_domain):
    try:
        f_in = h5py.File(stream,'r')
        load_h5file(f_in,destination_domain)
    except Exception as err:
        raise

def load_h5file(h5file,destination_domain):
    try:
        cfg = hsinfo.cfg
        fout = h5pyd.File(destination_domain, "w", endpoint=cfg["hs_endpoint"], username=cfg["hs_username"], password=cfg["hs_password"], bucket="charisma", retries=3, owner=cfg["hs_username"])
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
        print("create_chada_from_native",native_filename,chada_filename)

        create_chada_from_native(native_filename, chada_filename)
        load_h5file(h5py.File(chada_filename,'r'),destination_domain)
    except Exception as err:
        print(err)
        raise          
    finally:
        if native_filename!=None:
            os.remove(native_filename)
  

def load_domain(url):
    print(url)
    f = None
    try:
        url="/Round_Robin_1/LBF/S0P_WITEcAlpha532_100x_005_10000msx1.cha\\raw"
        f = h5pyd.File(url, 'r')
        return f.filename

    except Exception as err:
        raise err
    finally:
        if f!=None:
            f.close()