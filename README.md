# **CHARISMA - Raman spectrum harmonization**

## _This is the GIT development platform for algorithms that generate CHADA (characteristic data) from a given Raman spectrum. It refers to CHARISMA Work Package 4._

### Quick Start for users

#### Pip

***NOTE:** It is usually a good idea to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html).*
```
pip install -r requirements.txt
jupyter notebook
```

#### Conda
[Install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), then run the following:
```
conda env update -f environment.yml
conda activate ramanchada
jupyter notebook
```

#### Docker (on Linux)

***WARNING:** This will produce files in the `test` directory that will be owned by root. To fix it, do `sudo rm -r test`,
followed by `git checkout -- test`. An enhanced procedure without this problem may be introduced in the future.*

[Install Docker](https://docs.docker.com/get-docker/), then run the following:
```
docker pull python
docker run -it --rm -p 8888:8888 -v "${PWD}":/work python /bin/bash
```
In the container, run:
```
cd work
pip install -r requirements.txt
jupyter notebook --ip 0.0.0.0 --allow-root --no-browser
```

### Quick Start for contributors

Please see the [Contributor Guide](CONTRIBUTING.md).

### What is the CHARISMA Project ?
- Visit the [CHARISMA Home Page](https://www.h2020charisma.eu/) !

### What is CHADA ?
- _**CHA**racterization **DA**ta_

### ramanchada package

You have to install ramanchada package before using the notebooks - see _Quick Start_ above.

### Relevant use cases: what will CHADA be used for ?
[List of CHADA use cases](documents/Use cases CHADA.xlsx): _This is a living document - please add to it! Download, then `replace` by the changed doc._
* Use cases implemented in the current version of the CHADA library
    * [General demonstration of Chada class and data structure](https://gitlab.cc-asp.fraunhofer.de/barton/charisma-raman-spectrum-harmonization/-/blob/master/Chada%20new%20demo.ipynb)
    * [Demonstration of Chada generation from various text and CSV-based manufacturer formats](https://gitlab.cc-asp.fraunhofer.de/barton/charisma-raman-spectrum-harmonization/-/blob/master/Chada%20txt%20import%20demo.ipynb)
    * [Demonstration of CHADA Groups and their functions](https://gitlab.cc-asp.fraunhofer.de/barton/charisma-raman-spectrum-harmonization/-/blob/master/Chada%20Group%20demo.ipynb)
    * [Demonstration of CHADA wavenumber (x axis) calibration using a refernece spectrum](https://gitlab.cc-asp.fraunhofer.de/barton/charisma-raman-spectrum-harmonization/-/blob/master/Chada%20calibration%20demo.ipynb)

### Relevant Raman manufacturers, OEM software, and file formats
[List of relevant manufacturers and file formats](documents/Raman data formats.xlsx)

### CHADA data structure
[Description of CHADA data structure](documents/Charisma%20deliverable%204_1.docx)

### Chemometric methods for Raman data implemented in CHADA
[Chamometrics Review from CHARISMA WP3](documents/Charisma%20deliverable%204_1.docx)

### WWW resources (_Please add by entering [README.md](README.md) in `EDIT` mode!_)
#### Readers
- specio Python library for Spectroscopy I/O: https://specio.readthedocs.io/en/stable/sec_user.html
- spc file format: https://docuri.com/download/spc-file-format_59c1d322f581710b28653306_pdf
- _gwyddion_ file formats: http://gwyddion.net/documentation/user-guide-en/file-formats.html
- _wit_io_ repository (Matlab implementation of WITec SUITE): https://gitlab.com/jtholmi/wit_io
- Formats on _Spectroscopy Ninja_: https://www.effemm2.de/info/info_free_sw.html

#### Baseline separation
- Python code for _SNIP_: https://stackoverflow.com/questions/57350711/baseline-correction-for-spectroscopic-data, https://stackoverflow.com/questions/57350711/baseline-correction-for-spectroscopic-data
- Python code for ALS: https://stackoverflow.com/questions/29156532/python-baseline-correction-library

#### Spectrum calibration
- Affine transforms: https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_features.html
- _Rascal_ Python library for automated spectrometer wavelength calibration: https://pypi.org/project/rascal/, https://arxiv.org/abs/1912.05883
- _xcal_raman_ Python functions for wavenumber calibration of Raman spectrometers: https://pypi.org/project/xcal_raman/
- _Scikit-spectra_: Explorative Spectroscopy in Python: https://openresearchsoftware.metajnl.com/articles/10.5334/jors.bs/

---
🇪🇺 This project has received funding from the European Union’s Horizon 2020 research and innovation program under [grant agreement No. 952921](https://cordis.europa.eu/project/id/952921).
