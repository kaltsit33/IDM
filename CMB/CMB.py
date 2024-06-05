import camb
import numpy as np
import matplotlib.pyplot as plt

pars = camb.model.CAMBparams()
pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.122)
pars.set_IDM_params(O20=0.28, log_kC1=-2)
pars.InitPower.set_params(ns=0.965)
pars.set_for_lmax(2500)

results = camb.get_results(pars)

powers = results.get_cmb_power_spectra(pars)

totCL=powers['total']
ls = np.arange(totCL.shape[0])
plt.plot(ls,totCL[:,0], color='k')
plt.title(r'$TT\, [\mu K^2]$')
plt.show()