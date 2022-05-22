# + tags=["parameters"]
import uuid
import json
import h5pyd
from datetime import date
upstream = []
product = None
hsds_investigation = None
config_input = None
index_enabled_only = None
# -


def prefixed_uuid(value, prefix="CRMA"):
    return prefix+"-"+str(uuid.uuid3(uuid.NAMESPACE_OID, value))


class StudyRaman:
    def __init__(self, investigation, provider, parameters, filename):
        self.tags_params = ["instrument_s","wavelength_s" ]
        self.topcategory = "P-CHEM"
        self.endpointcategory = "ANALYTICAL_METHODS_SECTION"
        self.investigation = investigation
        self.method = "Raman spectroscopy"
        self.provider = provider
        self.parameters = parameters
        self.filename = filename


    def document_uuid(self,id,params):
        return prefixed_uuid("{} {} {} {}".format(self.investigation,self.provider,self.method,params))

    def to_solr_json(self):
        _solr = {}
        id = prefixed_uuid(self.filename)
        _solr["id"] = id
        _solr["investigation_uuid_s"] = prefixed_uuid(self.investigation)
        _solr["assay_uuid_s"] = prefixed_uuid(self.investigation)

        _solr["type_s"] = "study"


        _solr["topcategory_s"] = self.topcategory
        _solr["endpointcategory_s"] = self.endpointcategory
        _solr["guidance_s"] = "CHARISMA"
        _solr["guidance_synonym_ss"] = ["FIX_0000058"]
        _solr["E.method_synonym_ss"] = ["FIX_0000058"]
        _solr["endpoint_s"] = "Raman spectrum"
        _solr["effectendpoint_s"] = "RAMAN_CHADA_FILE"
        _solr["effectendpoint_synonym_ss"] = ["CHMO_0000823"]
        _solr["reference_owner_s"] = self.provider
        _solr["reference_year_s"] = date.today().strftime("%Y")
        _solr["reference_s"] = self.investigation
        _solr["textValue_s"] = self.filename
        _solr["updated_s"] = date.today().strftime("%Y-%m-%d")
        _solr["E.method_s"] = self.method

        _params = {}
        _conditions = {}


        for _prm in sorted(self.parameters):
            if _prm in self.tags_params:
                _params[_prm] = self.parameters[_prm]
            else:
                _conditions[_prm] = self.parameters[_prm]

        _solr["document_uuid_s"] = self.document_uuid(id,json.dumps(_params))

        _params["id"] = id + "/prm"
        _params["topcategory_s"] = self.topcategory
        _params["endpointcategory_s"] = self.endpointcategory
        _params["E.method_s"] = self.method
        _params["type_s"] = "params"


        _conditions["id"] = id + "/cn"
        _conditions["topcategory_s"] = self.topcategory
        _conditions["effectid_hs"] = id
        _conditions["endpointcategory_s"] = self.endpointcategory
        _conditions["type_s"] = "conditions"



        _conditions["document_uuid_s"] = _solr["document_uuid_s"]
        _params["document_uuid_s"] = _solr["document_uuid_s"]
        _solr["_childDocuments_"] = [_params,_conditions]

        return _solr


class Substance:
    def __init__(self, name, publicname, owner_name, substance_type=None):
        self.name = name
        self.publicname = publicname
        self.owner_name = owner_name
        self.substance_type = substance_type
        self.studies = []


    def add_study(self, study):
        self.studies.append(study)

    def to_solr_json(self):

        _solr = {}
        _solr["content_hss"] = [hsds_investigation]
        _solr["dbtag_hss"] = "CRMA"
        _solr["name_hs"] = self.name
        _solr["publicname_hs"] = self.publicname
        _solr["owner_name_hs"] = self.owner_name
        _solr["substanceType_hs"] = self.substance_type
        _solr["type_s"] = "substance"
        _suuid = prefixed_uuid(self.name)

        _solr["s_uuid_hs"] = _suuid
        _solr["id"] = _suuid
        _studies = []
        _solr["SUMMARY.RESULTS_hss"] = []
        for _study in self.studies:
            print(_study);
            _study_solr = _study.to_solr_json()
            _study_solr["s_uuid_s"] = _suuid
            _study_solr["type_s"] = "study"
            _study_solr["name_s"] = self.name
            _study_solr["publicname_s"] = self.publicname
            _study_solr["substanceType_s"] = self.substance_type
            _study_solr["owner_name_s"] = self.owner_name
            _studies.append(_study_solr)
            _summary = "{}.{}".format(
                _study.topcategory, _study.endpointcategory)
            if not (_summary in _solr["SUMMARY.RESULTS_hss"]):
                _solr["SUMMARY.RESULTS_hss"].append(_summary)
        _solr["_childDocuments_"] = _studies
        _solr["SUMMARY.REFS_hss"] = []
        _solr["SUMMARY.REFOWNERS_hss"] = []

        return _solr


def ramanchada2ambit(file_name,  substances={}, owner="CHARISMA"):
    h5 = h5pyd.File(file_name)
    _sample = h5["annotation_sample"].attrs["sample"]
    _tags = ["investigation", "provider"]
    _annotation_study = "annotation_study"
    _params = {}
    for attr in h5[_annotation_study].attrs:
        _attr = attr
        if attr in _tags:
            pass
        else:
            if _attr == "native_filename":
                _attr = "__input_file_s"
            else:
                _attr = attr + "_s"
            _params[_attr] = h5[_annotation_study].attrs[attr]

    if _sample not in substances:
        substances[_sample] = Substance(
            _sample, _sample, owner_name=owner, substance_type=lookup_substancetype(_sample))
    substances[_sample].add_study(StudyRaman(
        h5[_annotation_study].attrs["investigation"],
        h5[_annotation_study].attrs["provider"],
        _params, file_name))
    return substances


def lookup_substancetype(name):
    _dict = {
        "S0B": "CHEBI_30563",
        "S0N": "CHEBI_30563",
        "S1N": "CHEBI_30563",
        "S0P": "CHEBI_30563",
        "SIL": "CHEBI_30563",
        "NCAL": "CHEBI_46719",
        "SCAL": "CHEBI_46719",
        "PST": "CHEBI_61642"

    }
    try:
        return _dict[name.upper()]
    except Exception as err:
        return "CHEBI_59999"


substances = {}
# for _file in _files:
#    substances = ramanchada2ambit(_file, substances)

with open(config_input, 'r') as infile:
    config = json.load(infile)

for entry in config:

    if index_enabled_only and not entry["enabled"] :
       continue
    try:
        print(entry)
        h5domain = "/{}/{}/{}/{}/".format(hsds_investigation,
                                          entry["hsds_provider"], entry["hsds_instrument"], entry["hsds_wavelength"])
        domain = h5pyd.Folder(h5domain)
        _ndatasets = -1
        _r = 0
        n = domain._getSubdomains()
        if n > 0:
            for s in domain._subdomains:
                file_name = s["name"]
                if file_name.endswith(".cha"):
                    substances = ramanchada2ambit(file_name, substances)

                    pass
    except Exception as err:
        print(err)

with open(product["data"], 'w') as outfile:
    outfile.write("[")
    _sep = ""
    for key in substances:
        outfile.write(_sep)
        outfile.write(json.dumps(substances[key].to_solr_json()))
        _sep = ","
        outfile.flush()
    outfile.write("]")
