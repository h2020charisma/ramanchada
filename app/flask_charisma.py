from flask import Flask, request,jsonify
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast
from io import BytesIO
import uuid
import process
import traceback
import os

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

class ProcessDomain(Resource):

    def __init__(self):
        pass

    def get(self):
        domain = request.args.get('domain')
        try:
            name = process.load_domain(domain)
            return jsonify({'name': name})
        except Exception as err:
            traceback.print_exc()
            return {"err" : "xxx"}, 400 

    def post(self):

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
            
            return {"domain" : destination_domain, "filename" : f_name}, 200 
        except Exception as err:
            print(err)
            return {"err" : "xxx"}, 400 


              

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
api = Api(app)
#api.add_resource(Pipeline, '/pipeline')
#api.add_resource(Pipeline_dataset, '/pipeline/dataset')

api.add_resource(ProcessDomain, '/domain')

if __name__ == '__main__':
    app.run(debug=False)  # run our Flask app