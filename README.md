# **CHARISMA - Raman spectrum harmonization**

## _This is the GIT development platform for algorithms that generate CHADA (characteristic data) from a given Raman spectrum. It refers to CHARISMA Work Package 4._

### What is the CHARISMA Project ?
- Visit the [CHARISMA Home Page](https://www.h2020charisma.eu/) !

### What is CHADA ?
- _**CHA**racteristic **DA**ta_

### Relevant use cases: what will CHADA be used for ?
[List of CHADA use cases](documents/Use cases CHADA.xlsx): _This is a living document - please add to it! Download, then `replace` by the changed doc._
* Use cases implemented in the current version of the CHADA library
    * [General demonstration of Chada class and data structure](https://gitlab.cc-asp.fraunhofer.de/barton/charisma-raman-spectrum-harmonization/-/blob/master/Chada%20new%20demo.ipynb)
    * [Demonstration of CHADA Groups and their functions](https://gitlab.cc-asp.fraunhofer.de/barton/charisma-raman-spectrum-harmonization/-/blob/master/Chada%20Group%20demo.ipynb)
    * [Demonstration of CHADA wavenumber (x axis) calibration using a refernece spectrum](https://gitlab.cc-asp.fraunhofer.de/barton/charisma-raman-spectrum-harmonization/-/blob/master/Chada%20calibration%20demo.ipynb)

### Relevant Raman manufacturers, OEM software, and file formats
[List of relevant manufacturers and file formats](documents/Raman data formats.xlsx): _This is a living document - please add to it! Download, then `replace` by the changed doc._

### CHADA data structure
[Description of CHADA data structure](documents/Charisma%20deliverable%204_1.docx)

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
