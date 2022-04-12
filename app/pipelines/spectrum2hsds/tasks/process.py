# + tags=["parameters"]
upstream = []
product = None
hsds_investigation = None
config_input = None
dry_run = None
# -


from ramanchada.classes import RamanChada,SpectrumGroup
import matplotlib.pyplot as plt
import os,path
def peaks(h5dataset,fitmethod,RR,sample,debug=False):

    head,tail = os.path.split(h5dataset)
    print('#{} {} {}'.format(sample,head,tail))
    if dry_run:
        return RR    
    spec = RamanChada(h5dataset,raw=True,is_h5pyd=True)
    if debug:
        spec.plot()
        #C = RamanChada(h5dataset,raw=False,is_h5pyd=True)
        print('Correction of fluorescent background')
        print('Fit baseline model using the SNIP algorithm and remove from data')    
    spec.fit_baseline(method='snip')
    spec.remove_baseline()
    if debug:
        print('Reset data. Fit baseline model using the ALS algorithm and remove from data')
    spec.rewind(0)
    spec.fit_baseline(method='als')
    spec.remove_baseline()
    if debug:
        spec.plot()
        print('Correction of cosmic rays')
        print('Fit x ray model')
    spec.fit_xrays()
    if debug:
        spec.plot()
        plt.figure()
        plt.plot(spec.y, label='raw data')
        plt.plot(spec.xrays, label='x ray model')
        plt.legend()
        plt.show()
        print('Subtract x ray model and plot corrected spectrum')
    spec.remove_xrays()
    if debug:
        spec.plot()    

    G = SpectrumGroup([spec])
    if debug:
        print('Apply Savitzky-Golay smoothing filter')
    spec.smooth(method='sg')
    G.add(spec)
    G.process('x_crop', 0, 1500)
    if debug:
        G.plot()
        print('Fitting peaks for {}  with {}  function'.format(h5dataset,fitmethod))
    spec.normalize('minmax')
    spec.peaks(fitmethod=fitmethod,show=debug,interval_width=1.5)
    RR['sample'] = sample
    RR['domain'] = head
    RR['dataset'] = tail
    RR['Peak positions'] = spec.bands['position']
    RR[str(fitmethod)+' Intensity'] = spec.bands['intensity']
    RR[str(fitmethod)+' Prominence'] = spec.bands['prominence']
    RR[str(fitmethod)+' fitted position'] = spec.bands[str(fitmethod)+' fitted position']
    RR[str(fitmethod)+' FWHM'] = spec.bands[str(fitmethod)+' fitted FWHM']
    RR['FWHM'] = spec.bands['FWHM']
    RR[str(fitmethod)+' FWHM/FWHM'] = spec.bands[str(fitmethod)+' fitted FWHM']/spec.bands['FWHM']
    return RR
    spec.show_bands()    
#https://lmfit.github.io/lmfit-py/



import h5pyd
import pandas as pd
RR = {}
RRR = pd.DataFrame()


import json
with open(config_input, 'r') as infile:
    config = json.load(infile)
for entry in config:
    if not entry["enabled"]:    
        continue
    h5domain = "/{}/{}/{}/{}/".format(hsds_investigation,entry["hsds_provider"],entry["hsds_instrument"],entry["hsds_wavelength"])
    domain = h5pyd.Folder(h5domain)
    _ndatasets = -1
    _r = 0
    n = domain._getSubdomains()
    if n>0:
        for fitmethod in ['voigt','gl' ]:
            for s in domain._subdomains:
                file_name = s["name"]
                if file_name.endswith(".cha"):
                    _r = _r+1
                    h5 = h5pyd.File(file_name)
                    sample = h5["annotation_sample"].attrs["sample"]
                    _tag = "{}#{}".format(h5domain,sample)
                    if not _tag in RR:
                        RR[_tag] = pd.DataFrame()
                    try:
                        
                        _RR = peaks(file_name,fitmethod,RR[_tag],sample,debug=False)
                        RRR = RRR.append(_RR, ignore_index = True)
                    except Exception as err:
                        print(err)
                    
                if _ndatasets>0 and _r>_ndatasets:
                    break


if not os.path.exists(product["data"]):
    os.mkdir(product["data"])
RRR

RRR.to_csv(os.path.join(product["data"],"peaks.csv"),index=False)

RRR2= RRR.groupby(['domain','dataset','sample','Peak positions'],as_index=False).mean()
RRR2

RRR2.to_csv(os.path.join(product["data"],"peaks_mean.csv"),index=False)


