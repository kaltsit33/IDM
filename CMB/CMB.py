import camb
import numpy as np
import matplotlib.pyplot as plt

pars = camb.set_params(H0=67.5, O20=0.28, log_kC1=-2, 
                       ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                       As=2e-9, ns=0.965, halofit_version='mead', lmax=3000)

results = camb.get_results(pars)

powers = results.get_cmb_power_spectra(pars)

totCL=powers['total']
ls = np.arange(totCL.shape[0])
plt.plot(ls,totCL[:,0], color='k')
plt.title(r'$TT\, [\mu K^2]$')
plt.show()