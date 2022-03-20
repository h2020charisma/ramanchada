
# cd app
# python -m pytest  ..\tests\test_app.py::test_postmetadata

from study import StudyRegistration
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
            