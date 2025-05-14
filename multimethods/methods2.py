import numpy as np
import emcee
import scipy
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Prior values
H0 = 70.0
O20 = 0.3
log_kC1 = -5.0

### BAO
import BAO.result
def chi_square_BAO(log_kC1, O20, H0, rdh):
    return BAO.result.chi_square(log_kC1, O20, H0, rdh)

### SNe Ia
import SNe.result2
def chi_square_SNe(log_kC1, O20, H0):
    return SNe.result2.chi_square(log_kC1, O20, H0)

### MCMC

def lnlike(paras):
    O20, log_kC1, H0, rdh = paras
    chi2 = chi_square_BAO(log_kC1, O20, H0, rdh) + chi_square_SNe(log_kC1, O20, H0)
    return -0.5 * chi2

def lnprior(paras):
    O20, log_kC1, H0, rdh = paras
    if 0 < O20 < 0.5 and -10 < log_kC1 < 0 and 60 < H0 < 80 and 50 < rdh < 150:
        return 0.0
    return -np.inf

def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    initial = np.array([0.3, -5, 70, 100]) # expected best values
    pos = initial + 1e-4 * np.random.randn(50, 4)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 5000, progress = True)
    flat_samples = sampler.get_chain(discard=500, flat=True)
    np.savetxt('./multimethods/output2.dat', flat_samples)

if __name__ == '__main__':
    mp.freeze_support()
    main()
