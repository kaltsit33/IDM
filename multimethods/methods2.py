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

### OHD
import OHD.result 
def chi_square_OHD(log_kC1, O20, H0):
    return OHD.result.chi_square(log_kC1, O20, H0)

### SNe Ia
import SNe.result2
def chi_square_SNe(log_kC1, O20, H0):
    return SNe.result2.chi_square(log_kC1, O20, H0)

### MCMC

def lnlike(paras):
    O20, log_kC1, H0 = paras
    chi2 = chi_square_OHD(log_kC1, O20, H0) + chi_square_SNe(log_kC1, O20, H0)
    return -0.5 * chi2

def lnprior(paras):
    O20, log_kC1, H0 = paras
    if 0 < O20 < 0.5 and -10 < log_kC1 < 0 and 60 < H0 < 80 :
        return 0.0
    return -np.inf

def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.3, -5, 70]) # expected best values
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 5000, progress = True)
    flat_samples = sampler.get_chain(discard=500, flat=True)
    np.savetxt('./multimethods/output2.dat', flat_samples)

if __name__ == '__main__':
    mp.freeze_support()
    main()
