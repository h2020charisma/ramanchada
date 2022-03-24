
# cd app
# python -m pytest  ..\tests\test_app.py::test_postmetadata

from study import StudyRegistration, ParamsRaman
from process5 import delete_domain_recursive

def test_postmetadata():
    data = { "investigation" : "SANDBOX", "provider" : "IDEA",
                "instrument" : {"brand" : "TEST", "model" : "MODEL", "wavelength" : 785 }}
    _domain = "/SANDBOX/IDEA/TEST_MODEL/785/"

    try:
        delete_domain_recursive(_domain)
    except Exception as err:
        pass

    try:
        sr = StudyRegistration();
        domain = sr.post_metadata(data[ "investigation"],data["provider"],data["instrument"],"metadata.h5",True,["investigation","provider","instrument","wavelength"]);

        assert domain == _domain

    except Exception as err:
        assert False

def test_putmetadata():
    metadata = {
                "instrument" : { "optical_components" : {}}
                }
    op1 = { "id" : "OP1"}
    for _key in [ParamsRaman.COLLECTION_OPTICS.value,ParamsRaman.GRATINGS.value,ParamsRaman.SLIT_SIZE.value,
            ParamsRaman.PIN_HOLE_SIZE.value,ParamsRaman.COLLECTION_FIBRE_DIAMETER.value,ParamsRaman.OTHER.value
            ]:
        metadata["instrument"][_key] = [_key]
        op1[_key] = _key
    op1["laser_power"] =  [
                {
                "settings": 100,
                "power_mw": 11
                }
                ]
    metadata["instrument"]["optical_paths"] = [ op1  ]

    _domain = "/SANDBOX/IDEA/TEST_MODEL/785/"

    try:
        sr = StudyRegistration();
        domain =  sr.put_metadata(_domain,metadata,"all","metadata.h5")
        assert domain == _domain

    except Exception as err:
        assert False
