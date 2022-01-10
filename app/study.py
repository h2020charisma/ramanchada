import h5pyd

from enum import Enum

import traceback
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
    LASER_POWER = "laser_power"
    SETTINGS = "settings"
    POWER_MW = "power_mw"

class StudyRegistration:
    def __init__(self):
        self._tag_instrument = "instrument"
        self._tag_optical_paths = "optical_paths"
        self._tag_laser_power = "laser_power"
        self._tag_id = "id"
        self.keys  = [ParamsRaman.COLLECTION_OPTICS.value,ParamsRaman.GRATINGS.value,ParamsRaman.SLIT_SIZE.value,
            ParamsRaman.PIN_HOLE_SIZE.value,ParamsRaman.COLLECTION_FIBRE_DIAMETER.value,ParamsRaman.OTHER.value
            ]

        pass
           

    def update_opticalpaths(self,metadata,_out):
        with h5pyd.File(_out  ,"r+") as f:
            try:
                self.opticalpaths2h5(metadata["instrument"],f["instrument"])
            except Exception as err:
                raise err    

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

    def opticalcomponents2h5(self,instrument,h5_group):
        for key in self.keys:
            try:
                h5_group.attrs[key] = instrument[key]
            except:
               #delete if existing
                h5_group.attrs[key] = []
           

    def opticalpaths2h5(self,instrument,h5_group):
        print(">>",instrument)
        try:
            _g_ops = h5_group.require_group(self._tag_optical_paths)
            #remove optical paths which are not in the current list
            for g in h5_group[self._tag_optical_paths]:
                if not (g in instrument[self._tag_optical_paths]):
                    del h5_group[self._tag_optical_paths][g] 
        except Exception as err:
            print(err)
        for op in instrument[self._tag_optical_paths]:
            try:
                _g_op = _g_ops.require_group(op[self._tag_id])
                for key in self.keys:
                    _g_op.attrs[key] = op[key]                
            except Exception as err:
                print(op,err)

            for key in _g_op.keys():
                del _g_op[key]
            for laser_power_measurement in op[ParamsRaman.LASER_POWER.value]:
                try:
                    _g_laserpower = _g_op.require_group("{}_{}".format(ParamsRaman.LASER_POWER.value,laser_power_measurement[ParamsRaman.SETTINGS.value]))
                    _g_laserpower.attrs[ParamsRaman.SETTINGS.value] = laser_power_measurement[ParamsRaman.SETTINGS.value]
                    try:
                        if ParamsRaman.POWER_MW.value in laser_power_measurement:
                            _g_laserpower.attrs[ParamsRaman.POWER_MW.value] = laser_power_measurement[ParamsRaman.POWER_MW.value]
                    except:
                        _g_laserpower.attrs[ParamsRaman.POWER_MW.value] = ""
                except Exception as err:
                    print(traceback.format_exc())

    
    

    def instrument2h5(self,instrument,h5_group):
        h5_group.attrs[self._tag_id] = self.get_instrument_id(instrument)
        h5_group.attrs[ParamsRaman.BRAND.value] = instrument[ParamsRaman.BRAND.value]
        h5_group.attrs[ParamsRaman.MODEL.value] = instrument[ParamsRaman.MODEL.value]
        h5_group.attrs[ParamsRaman.WAVELENGTH.value] = instrument[ParamsRaman.WAVELENGTH.value]
        self.opticalcomponents2h5(instrument,h5_group)  
                #print(instrument["optical_paths"]) 
        self.opticalpaths2h5(instrument,h5_group);



    def metadata2h5(self,metadata,h5file): 
        h5file.attrs["investigation"] = metadata["investigation"]
        h5file.attrs["provider"] = metadata["provider"]
        try:
            _g_instrument = h5file.require_group(self._tag_instrument)
            instrument = metadata[self._tag_instrument]
            self.instrument2h5(instrument,_g_instrument)
        except Exception as err:
            print(err)

    def h52metadata(self,h5file): 
        metadata = {}
        metadata["investigation"] = h5file.attrs["investigation"]
        metadata["provider"] = h5file.attrs["provider"]
        group_instrument = h5file[self._tag_instrument]
        try:
            instrument = self.h52instrument(group_instrument)
            metadata[self._tag_instrument] = instrument
        except Exception as err:
            print(err)
        
                
        return metadata

    def h52instrument(self,group_instrument):
        optical_paths = []
    
        instrument = {
            self._tag_id : group_instrument.attrs[self._tag_id],
            ParamsRaman.BRAND.value : group_instrument.attrs[ParamsRaman.BRAND.value],
            ParamsRaman.MODEL.value : group_instrument.attrs[ParamsRaman.MODEL.value],
            self._tag_optical_paths : optical_paths
        }
        for key in self.keys:
            try:
                instrument[key] =  group_instrument.attrs[key].tolist() ;
            except Exception as err:
                instrument[key] = None
        try:
            _g_ops= group_instrument[self._tag_optical_paths]
            
            for op in _g_ops.keys():
                print(op);
                
                _g_op=_g_ops[op]
                optical_path = {}
                optical_path["id"] = op
                optical_paths.append(optical_path) 
                for key in self.keys:
                    try:
                        if isinstance(_g_op.attrs[key],str):
                            optical_path[key] = _g_op.attrs[key]
                        else:
                            optical_path[key] = _g_op.attrs[key].tolist()
                    except Exception as err:
                        print(err)                        
                optical_path[ParamsRaman.LASER_POWER.value] = [] 
                for laser_power_measurement in _g_op.keys():
                    tmp = {}
                    tmp[ParamsRaman.SETTINGS.value] = _g_op[laser_power_measurement].attrs[ParamsRaman.SETTINGS.value].tolist()
                    try:
                        tmp[ParamsRaman.POWER_MW.value] = _g_op[laser_power_measurement].attrs[ParamsRaman.POWER_MW.value].tolist()
                    except:
                        tmp[ParamsRaman.POWER_MW.value] = None
                    optical_path[ParamsRaman.LASER_POWER.value].append(tmp)

        except Exception as err:
            print(err)
        return instrument                         


                  