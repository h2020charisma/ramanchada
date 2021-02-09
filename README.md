# **CHARISMA - Raman spectrum harmonization**

## _This is the GIT development platform for algorithms that generate CHADA (characteristic data) from a given Raman spectrum. It refers to CHARISMA Work Package 4._

### Relevant use cases: what will CHADA be used for ?
[List of CHADA use cases](documents/Use cases CHADA.xlsx): This is a living document - please add to it! Download, then `replace` by the changed doc.

### Relevant Raman manufacturers, OEM software, and file formats
[List of relevant manufacturers and file formats](documents/Raman data formats.xlsx): This is a living document - please add to it! Download, then `replace` by the changed doc.


### WWW resources (_Please add!_)
#### Readers
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

### CHADA meta data seed list	
- Instrument	
1. Manufacturer	WITec Instruments, Ulm, Germany
1. Model	Alpha 500
1. Type	Confocal Raman Microscope
1. System ID:	130-1200-830
1. Location:	Darmstadt, Germany, 49.9, 8.66

- General:	
1. Time:	4:47:58 PM
1. Date:	Monday, November 30, 2020
1. User Name:	Bastian
1. Sample Name:	SL graphene on TEM grid
1. Configuration:	Raman 785

- Spectrometer	
1. Name	UHTS300:
1. Excitation Wavelength [nm]:	785
1. Grating:	G1: 600 g/mm BLZ=750nm
1. Center Wavelength [nm]:	847.998
1. Spectral Center [rel. 1/cm]:	946.372
__ 	
- CCD camera	
1. Name	DU401_DD:
1. Width [Pixels]:	1024
1. Height [Pixels]:	128
1. Temperature [°C]:	-59
1. Integration Time [s]:	5.01222
1. Camera Serial Nr.:	14907
1. AD Converters:	AD1 (16Bit)
1. Vertical Shift Speed [µs]:	[8.25]
1. Horizontal Shift Speed [MHz]:	0.1
1. Preamplifier Gain:	1
1. ReadMode:	Full Vertical Binning

- Acquisition parameters	
1. Excitation Wavelength [nm]:	785
1. Laser power [mW]:	5
1. Number Of Accumulations:	1
1. Integration Time [s]:	5.01222

- Optics	
1. Objective:	
1. Objective Name:	Zeiss LD EC Epiplan-Neofluar 50x / 0.55
1. Objective N.A.	0.55
1. Objective Magnification:	50
1. Other	

