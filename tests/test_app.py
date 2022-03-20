
# cd app
# python -m pytest  ..\tests\test_app.py::test_postmetadata

from study import StudyRegistration

def test_postmetadata():
    data = { "investigation" : "SANDBOX", "provider" : "IDEA",
                "instrument" : {"brand" : "TEST", "model" : "MODEL", "wavelength" : 785 }}

    try:
        sr = StudyRegistration();
        domain = sr.post_metadata(data[ "investigation"],data["provider"],data["instrument"],"metadata.h5",True,["investigation","provider","instrument","wavelength"]);

        assert domain == "/SANDBOX/IDEA/TEST_MODEL/785/"
                
    except Exception as err:
        assert False
            