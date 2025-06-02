import numpy as np
import scipy
import multiprocessing as mp
import emcee

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from caculus import chi_square_OHD
from caculus import chi_square_SNe
from caculus import chi_square_BAO

def lnlike(paras):
    O20, n, H0, rdh = paras
    chi2 = chi_square_OHD(O20, n, H0) + chi_square_SNe(O20, n, H0) + chi_square_BAO(O20, n, H0, rdh)
    return -0.5 * chi2

def lnprior(paras):
    O20, n, H0, rdh = paras
    if 0 < O20 < 1 and -10 < n < 10 and 50 < H0 < 100 and 50 < rdh < 150:
        return 0.0
    return -np.inf

def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.3, 0, 70, 100]) # expected best values
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 4)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 5000, progress = True)

    flat_samples = sampler.get_chain(discard=500, flat=True)
    np.savetxt('./wIDM/output/m3.dat', flat_samples)

if __name__ == "__main__":
    mp.freeze_support()
    main()
