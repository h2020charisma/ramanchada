# + tags=["parameters"]
upstream = None
product = None
folder_native = None
# -

from processors import FileProcessor
#,parse_name
import json
import logging,sys,os

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class MetadataProcessor:
    def __init__(self,file_output,lookup={"brands":[]}):
        self.file_output=file_output
        self.set_lookup(lookup)
        self.metadata={}

    def set_lookup(self,lookup={}):
        self.lookup=lookup

    def parse_name(self,f_name):
        filename, file_extension = os.path.splitext(f_name)
        if file_extension==".xlsx" or file_extension==".json": 
            return None  
        tags = os.path.basename(filename).split("_") 
        #logger.debug(f_name,len(tags),tags)
        tag_time = 4
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

        for b in self.lookup["brands"]:
            if instrument.startswith(b):
                instrument_brand = b
                instrument_model = instrument.replace(b,"")
            break;  
        results = {
            "source" : {
                "basename" : os.path.basename(filename),
                "extension" : file_extension
            },
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

    def write(self):
        with open(self.file_output, 'w') as outfile:
            json.dump(self.metadata, outfile)                  


FP = FileProcessor(folder_input=folder_native,folder_output=None)
MP = MetadataProcessor(file_output=product["metadata"])

def gather_metadata(f_name):
    print(f_name)
    try:
        logger.info(f_name)
        tmp = MP.parse_name(f_name)
        print(tmp)
        if tmp !=None:
            key = tmp["source"]["basename"]
            MP.metadata[key] = tmp
    except Exception as err:
        print(f_name,err)
    

FP.process_file(folder_native,recurse=True,callback=gather_metadata)
MP.write()

