from flask import Flask, request,jsonify
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast
from io import BytesIO
import uuid
import process
import traceback
import os,sys
import json
from werkzeug.exceptions import BadRequest, BadGateway, NotFound, HTTPException

from process import TaskResult, TaskStatus
from study import ParamsRaman, StudyRegistration

class Pipeline(Resource):
    #tbd load from config

    def __init__(self,_pipelines={ "dataset" : {} }):
        self.pipelines=_pipelines

    def get(self):
        return self.pipelines, 200 

class Pipeline_dataset(Pipeline) :
    #tbd load from config
    def __init__(self,_pipelines={ "normalize" : {}, "baseline" : {} }):
        super().__init__(_pipelines)

# all Ramanchada functions
class AlgorithmResource(Resource):
    def __init__(self):
        pass

    def run(self,R, algorithm):
        if "normalise" == algorithm:
            R.normalize()
            R.commit(algorithm)
            return R
        elif "baseline" == algorithm:
            R.baseline()
            R.commit(algorithm)    
            return R
        elif "smooth" == algorithm:
            R.smooth()
            R.commit(algorithm)
            return R
        else:
            return None                      

    def post(self):
        tr = TaskResult(name="algorithm")
        paths = ["dataset","algorithm"]
        params = {}
      
        for p in paths:
            params[p] = None
            try:
                params[p]  = request.form[p]
            except:
                tr.set_error("Missing {}".format(p))
                return tr.to_dict(), 400          

        try:
            R = process.load_domain(params["dataset"],True)
            self.run(R,params["algorithm"])

            tr.set_completed(params["dataset"])
            return tr.to_dict()  ,200
        except Exception as err:
            (type, value, traceback) = sys.exc_info()
            tr.set_error(str(value),str(type))
            return tr.to_dict(),400

import h5pyd 

class ProcessDomainResource(Resource):

    def __init__(self):
        self.paths = ["investigation","provider","instrument","wavelength","optical_path","sample","laser_power"]
        self.create_folders = False
        
        pass

    def read_file(self,domain,result):
        with process.read_file(domain) as file:
            tmp,datasets = process.get_file_annotations(file)

            result["annotation"].append(tmp)  
            result["datasets"] = datasets
        return result

    def get(self):
        tr = None
        try:
            domain = request.args.get('domain')
            tr = TaskResult(name=domain)
        except:
            return {"err" : "domain parameter missing"}, 400
        try:
            raw = request.args.get('raw')
        except:
            raw=False
        try:    
            result = { "subdomains" : [],"domain" : domain, "annotation" : [], "datasets" : []}
            if domain is None:
                tr.set_error("missing domain")
                return tr.to_dict(),400
            if domain.endswith("/"):
                with process.check_folder(domain,create=False) as folder:
                    n = folder._getSubdomains()

                    for s in folder._subdomains:
                        subdomain = {"name" : s["name"], "annotation" : [], "last_segment" : s["name"].split("/")[-1]}
                        
                        try:
                            with process.read_file(s["name"]) as file:
                                tmp,datasets = process.get_file_annotations(file)
                                subdomain["annotation"].append(tmp) 
                                subdomain["datasets"] = datasets
                                pass
                        except:
                            pass                       
                        result["subdomains"].append(subdomain)

                return  result,  200                
            else:
                self.read_file(domain,result)
       
                return  result,  200 
        except Exception as err:
            (type, value, traceback) = sys.exc_info()
            tr.set_error(str(value),str(type))
            return tr.to_dict(),400

    def check_paths(self,params,paths,skip_paths): 

        #print(params)
        folder = ""
        h5folder=None
        for p in paths:
            if p in skip_paths:
                continue;
            folder = "{}/{}".format(folder,params[p])
            domain="{}/".format(folder)
            try:
                h5folder = process.check_folder(domain,create=self.create_folders)
                h5folder.close()
            except IOError as err:
                if err.errno==NotFound.code:
                    raise NotFound(domain + " " + str(err))
                else:
                    raise BadGateway(domain + " " + str(err),err.errno)
            except Exception as err:
                raise BadGateway(domain + " " + str(err))

                    
        return domain,folder;
    
# curl http://127.0.0.1:5000/dataset -F "file=@PST10_iR532_Probe_005_3000msx7.spc" -F "provider=FNMT-Madrid" -F "investigation=Round_Robin_1" -F "instrument=BWTek" -F "wavelength=532" -F "optical_path=Probe" -F "sample=PST10" -u user
# {"name": "POST /domain", "result": "/Round_Robin_1/FNMT-Madrid/BWTek/532/Probe/PST10_iR532_Probe_005_3000msx7.cha", "error": null, "errorCause": null, "completed": "Mon Dec  6 16:14:21 2021", "started": "Mon Dec  6 16:14:21 2021", "status": "TaskStatus.Completed"}
    def post(self):
        tr = TaskResult("POST /dataset")
        skip_paths=["sample","laser_power"]
        params = {}
        domain = None
        folder  = None
        try:
            params = {} 
      
            for p in self.paths:
                params[p] = None
                try:
                    params[p]  = request.form[p]
                except:
                    raise BadRequest("Missing {}".format(p))

            domain,folder = self.check_paths(params,self.paths,skip_paths)
        except HTTPException as err:
            tr.set_error(str(err))
            return tr.to_dict(), err.code   
        except Exception as err:
            print("!!!!",str(err))
            tr.set_error(str(err))
            return tr.to_dict(), BadRequest.code   
      
         
        try:
            uploaded_file = request.files['file']
            f_name = uploaded_file.filename
            if f_name==None or f_name=="":
                tr.set_error("Missing file")
                #print(tr.to_dict());
                return tr.to_dict(), BadRequest.code 
        except Exception as err:
            (type, value, traceback) = sys.exc_info()
            tr.set_error(str(value),str(type))
            return tr.to_dict(), BadRequest.code          


        try:
            filename, file_extension = os.path.splitext(f_name)

            if file_extension==".cha":
                destination_domain="{}/{}".format(folder,f_name)
                process.load_h5stream(uploaded_file.stream,destination_domain,params)
            else:
                destination_domain="{}/{}.cha".format(folder,filename)
                process.load_native(file=uploaded_file,f_name=f_name,destination_domain=destination_domain,
                    params=params)
            
            tr.set_completed(destination_domain)
            return tr.to_dict()  ,200
        except HTTPException as err:
            tr.set_error(str(err))
            return tr.to_dict(), err.code           
        except Exception as err:
            print(err);
            (type, value, traceback) = sys.exc_info()
            tr.set_error(str(value),str(type))
            print(tr.to_dict());
            return tr.to_dict(), 400 

from flask.json import JSONEncoder

class StudyRegistrationResource(ProcessDomainResource):
    
    def __init__(self):
        super(StudyRegistrationResource).__init__()
        self.paths = ["investigation","provider","instrument","wavelength"]
        self.create_folders = True
        self.METADATA_FILE = "metadata.h5"
        
    def read_file(self,domain,result):
        if domain.endswith(self.METADATA_FILE):
            sr = StudyRegistration();
            with process.read_file(domain) as file:
                metadata = sr.h52metadata(file)
                result["metadata"] = metadata
            return result
        else:
            super().read_file(domain,result)

    def get(self):
        return super().get();

    def put(self):
        tr = TaskResult("PUT /metadata")
        
        params = {}
        
        sr = StudyRegistration();
        try:
            domain = request.json["domain"]  
            metadata = request.json["metadata"]
            mode = request.json["mode"] # optical_components | optical_paths
            print(request.json)
            if domain.endswith(self.METADATA_FILE):
                _out=domain
            else:
                _out = "{}{}".format(domain,self.METADATA_FILE)

            if mode=="optical_components":
                with h5pyd.File(_out  ,"r+") as f: 
                    if len(f["instrument"]["optical_paths"].keys())==0:
                        sr.opticalcomponents2h5(metadata["instrument"],f["instrument"])
                    else:
                       tr.set_error("Can't modify components; remove all optical paths first.") 
                       return tr.to_dict(), BadRequest.code  
            elif mode=="optical_paths": 
                with h5pyd.File(_out  ,"r+") as f:
                    try:
                        sr.opticalpaths2h5(metadata["instrument"],f["instrument"])
                    except Exception as err:
                        print(err)

            tr.set_completed(domain)
            return tr.to_dict(), 200 
        except HTTPException as err:
            print(err)
            tr.set_error(str(err))
            return tr.to_dict(), err.code                 
        except Exception as err:
            tr.set_error(str(err))
            return tr.to_dict(), BadRequest.code    

    def post(self):
        tr = TaskResult("POST /metadata")
        
        params = {}
        #print(request.json);
        sr = StudyRegistration();
        try:
            params["provider"] = request.json["provider"]  
            params["investigation"] = request.json["investigation"]     
            _tag_instrument = request.json["instrument"]            
            metadata = sr.create_metadata(
                params["investigation"],params["provider"],
                brand = _tag_instrument[ParamsRaman.BRAND.value],
                model=_tag_instrument[ParamsRaman.MODEL.value],
                wavelength=_tag_instrument[ParamsRaman.WAVELENGTH.value]);
            params["instrument"] = sr.get_instrument_id(metadata["instrument"])
            params[ParamsRaman.WAVELENGTH.value] = metadata["instrument"][ParamsRaman.WAVELENGTH.value]
            
            domain,folder = self.check_paths(params,self.paths,skip_paths=[])
            _out = "{}{}".format(domain,self.METADATA_FILE)
            print(metadata)
            with h5pyd.File(_out  ,"w") as f:
                sr.metadata2h5(metadata,f)

            tr.set_completed(domain)
            return tr.to_dict(), 200 
        except HTTPException as err:
            print(err)
            tr.set_error(str(err))
            return tr.to_dict(), err.code                 
        except Exception as err:
            tr.set_error(str(err))
            return tr.to_dict(), BadRequest.code    

        
        
            


app = Flask(__name__)

from werkzeug.exceptions import BadGateway, BadRequest, HTTPException

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
api = Api(app)
#api.add_resource(Pipeline, '/pipeline')
#api.add_resource(Pipeline_dataset, '/pipeline/dataset')

api.add_resource(ProcessDomainResource, '/dataset')
api.add_resource(AlgorithmResource, '/algorithm')
api.add_resource(StudyRegistrationResource, '/metadata')
if __name__ == '__main__':
    app.run(debug=True)  # run our Flask app