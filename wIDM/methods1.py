import numpy as np
import scipy
import matplotlib.pyplot as plt
import multiprocessing as mp
import emcee
from getdist import plots, MCSamples

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from caculus import chi_square_QSO
from caculus import chi_square_SNe


def lnlike(paras):
    O20, n, H0, gamma0, gamma1, beta0, beta1, delta = paras
    chi2 = chi_square_QSO(O20, n, H0, gamma0, gamma1, beta0, beta1, delta) + chi_square_SNe(O20, n, H0)
    return -0.5 * chi2

def lnprior(paras):
    O20, n, H0, gamma0, gamma1, beta0, beta1, delta = paras
    if 0 < O20 < 1 and -10 < n < 10 and 50 < H0 < 100 and 0<gamma0<1 and -1<gamma1<1 and -15<beta0<15 and -10<beta1<10 and 0<delta<1:
        return 0.0
    return -np.inf


def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.3, 0, 70, 0.6, 0, 6, 1, 0.2])
    # soln = scipy.optimize.minimize(nll, initial)
    # pos = soln.x + 1e-4 * np.random.randn(50, 3)
    pos = initial + 1e-4 * np.random.randn(50, 8)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 5000, progress = True)

    flat_samples = sampler.get_chain(discard=500, flat=True)
    np.savetxt('./wIDM/output/m1.dat', flat_samples)
