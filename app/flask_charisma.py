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

    def post(self):
        pass


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

    def post(self):
        tr = TaskResult("POST /domain")
        uploaded_file = request.files['file']
        
        provider = None
        try:
            provider = request.form['provider']
        except:
            tr.set_error("Missing provider")
            return tr.to_dict(), 400 
            
        folder = "/Round_Robin_1/{}/".format(provider)        
        f_name = uploaded_file.filename
        if f_name==None or f_name=="":
            tr.set_error("Missing file")
            return tr.to_dict(), 400 
        try:
            filename, file_extension = os.path.splitext(f_name)
            
            process.check_folder(folder)

            if file_extension==".cha":
                destination_domain="{}/{}".format(folder,f_name)
                process.load_h5stream(uploaded_file.stream,destination_domain)
            else:
                destination_domain="{}/{}.cha".format(folder,filename)
                process.load_native(uploaded_file,f_name,destination_domain)
            
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

api.add_resource(ProcessDomain, '/domain')

if __name__ == '__main__':
    app.run(debug=False)  # run our Flask app