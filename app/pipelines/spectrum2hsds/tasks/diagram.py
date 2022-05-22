# + tags=["parameters"]
upstream = []
product = None

# -



from diagrams import Cluster, Diagram
from diagrams.generic.storage import Storage
from diagrams.programming.framework import Flask
from diagrams.programming.language import Python, Javascript
from diagrams.custom import Custom
from diagrams.onprem.search import Solr
from diagrams.programming.flowchart import ManualInput

import os.path

with Diagram("CHARISMA Data repository", show=False, filename=os.path.join(product["data"],"diagram")):

    with Cluster("Data and metadata sources"):
        source = [
        Custom("CHARISMA Wikidata", os.path.abspath("resources/logo_wikidata.png")),
        Storage("Shared drive"),Storage("Shared drive"),
        Custom("Nomad-lab", os.path.abspath("resources/logo_nomad.png"))
        ]


    with Cluster("Ingestion and transformation"):
        processing =  Custom("Ploomber workflow (batch upload)", os.path.abspath("resources/logo_ploomber.png"))
        ramanchada = Python("ramanchada")
        flask = Custom("CHARISMA API", os.path.abspath("resources/logo_flaskrestful.png"))
        #Flask("CHARISMA API")
        processing - ramanchada - flask
        #processing = [Python("ramanchada"), Python("ramanchada2")]


    with Cluster("Storage"):
        hsds = Custom("HSDS", os.path.abspath("resources/logo_HDF.png"))
        storage_charisma = [hsds]


    with Cluster("Query and Processing"):
        with Cluster("Query engine"):
            query =  Custom("Ploomber workflow (indexing)", os.path.abspath("resources/logo_ploomber.png"))
            solr = Solr("SOLR search engine")
            query - solr
        with Cluster("Data science"):
            datascience = Python("ramanchada")
            datascience - Custom("Jupyter notebooks", os.path.abspath("resources/logo_jupyter.png"))
        #processing - Python("ramanchada")

    with Cluster("Output"):
        with Cluster("CHARISMA Web app"):
            js =  Javascript("JavaScript")
            app = js - Custom("11ty framework", os.path.abspath("resources/logo_11ty.png"))


    ManualInput("Manual input") >> js

    source >> processing
    flask >> storage_charisma
    storage_charisma >> flask
    flask >> datascience
    flask >> app
    solr >> app >> flask
    hsds >> query
    #solr >> app



