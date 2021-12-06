from flask import Flask, request,jsonify
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast
from io import BytesIO
import uuid
import process
import traceback
import os
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
            tr.set_error(traceback.print_exc(err))
            #traceback.print_exc()
            #print(">",traceback.format_exc(err))
            #return jsonify({"err" : str(traceback.format_exc(err))}),400
            return tr.to_dict(),400

    def post(self):
        tr = TaskResult("POST /domain")
        uploaded_file = request.files['file']
        
        provider = "UNKNOWN"
        try:
            provider = request.form['provider']
        except:
            pass
        f_name = uploaded_file.filename
        if f_name==None or f_name=="":
            return {"err" : "No file"}, 400 
        try:
            filename, file_extension = os.path.splitext(f_name)
            
            if file_extension==".cha":
                destination_domain="/Round_Robin_1/{}/{}".format(provider,f_name)
                process.load_h5stream(uploaded_file.stream,destination_domain)
            else:
                destination_domain="/Round_Robin_1/{}/{}.cha".format(provider,filename)
                process.load_native(uploaded_file,f_name,destination_domain)
            
            tr.set_completed(destination_domain)
            return tr.to_dict()  ,200
        except Exception as err:
            tr.set_error("xx","yy")
            traceback.print_exc()
            return tr.to_dict(), 400 

from flask.json import JSONEncoder


app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
api = Api(app)
#api.add_resource(Pipeline, '/pipeline')
#api.add_resource(Pipeline_dataset, '/pipeline/dataset')

api.add_resource(ProcessDomain, '/domain')

if __name__ == '__main__':
    app.run(debug=False)  # run our Flask app