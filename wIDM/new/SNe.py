import numpy as np
import multiprocessing as mp
import emcee

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from caculus import chi_square_SNe

def lnlike(paras):
    log_kC1, O20, n, H0 = paras
    chi2 = chi_square_SNe(log_kC1, O20, n, H0)
    return -0.5 * chi2

def lnprior(paras):
    log_kC1, O20, n, H0 = paras
    if -10 < log_kC1 < 0 and 0 < O20 < 1 and -5 < n < 10 and 50 < H0 < 100:
        return 0.0
    return -np.inf

def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    initial = np.array([-5.0, 0.3, 0, 70]) # expected best values
    pos = initial + 1e-4 * np.random.randn(50, 4)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 5000, progress = True)

    flat_samples = sampler.get_chain(discard=500, flat=True)
    np.savetxt('./wIDM/new/output/sne.dat', flat_samples)  

if __name__ == "__main__":
    mp.freeze_support()
    main()