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


from process import TaskResult, TaskStatus

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

class Algorithm(Resource):
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

class ProcessDomain(Resource):

    def __init__(self):
        pass


    def get(self):
        tr = TaskResult(name="domain")
        try:
            domain = request.args.get('domain')
        except:
            return {"err" : "domain parameter missing"}, 400
        try:
            raw = request.args.get('raw')
        except:
            raw=False
        try:
            R = process.load_domain(domain,raw)
            #return jsonpickle.encode(R, unpicklable=False),200
            tr.set_completed(domain)
            return R.meta  ,200
        except Exception as err:
            (type, value, traceback) = sys.exc_info()
            tr.set_error(str(value),str(type))
            return tr.to_dict(),400

# curl http://127.0.0.1:5000/dataset -F "file=@PST10_iR532_Probe_005_3000msx7.spc" -F "provider=FNMT-Madrid" -F "investigation=Round_Robin_1" -F "instrument=BWTek" -F "wavelength=532" -F "optical_path=Probe" -F "sample=PST10" -u user
# {"name": "POST /domain", "result": "/Round_Robin_1/FNMT-Madrid/BWTek/532/Probe/PST10_iR532_Probe_005_3000msx7.cha", "error": null, "errorCause": null, "completed": "Mon Dec  6 16:14:21 2021", "started": "Mon Dec  6 16:14:21 2021", "status": "TaskStatus.Completed"}
    def post(self):
        tr = TaskResult("POST /domain")
        uploaded_file = request.files['file']
        
        paths = ["investigation","provider","instrument","wavelength","optical_path","sample"]
        params = {}
      
        for p in paths:
            params[p] = None
            try:
                params[p]  = request.form[p]
            except:
                tr.set_error("Missing {}".format(p))
                return tr.to_dict(), 400   
        folder = ""
        for p in paths:
            if p!="sample":
                folder = "{}/{}".format(folder,params[p])
                domain="{}/".format(folder)
                h5folder = process.check_folder(domain,create=True)
                h5folder.close()
              
        #tr.set_error(folder)
        #return tr.to_dict(), 400     
         
        f_name = uploaded_file.filename
        if f_name==None or f_name=="":
            tr.set_error("Missing file")
            return tr.to_dict(), 400 
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

        except Exception as err:
            (type, value, traceback) = sys.exc_info()
            tr.set_error(str(value),str(type))
            return tr.to_dict(), 400 

from flask.json import JSONEncoder


app = Flask(__name__)

from werkzeug.exceptions import HTTPException

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

api.add_resource(ProcessDomain, '/dataset')
api.add_resource(Algorithm, '/algorithm')

if __name__ == '__main__':
    app.run(debug=False)  # run our Flask app