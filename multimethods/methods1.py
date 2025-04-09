import numpy as np
import emcee
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

H0 = 70.0
O20 = 0.3
log_kC1 = -5.0

import QSO.result2
def chi_square_QSO(log_kC1, O20, H0, gamma0, gamma1, beta0, beta1, delta):
    return QSO.result2.chi_square(log_kC1, O20, H0, gamma0, gamma1, beta0, beta1, delta)

import SNe.result2
def chi_square_SNe(log_kC1, O20, H0):
    return SNe.result2.chi_square(log_kC1, O20, H0)

def lnlike(paras):
    O20, log_kC1, H0, gamma0, gamma1, beta0, beta1, delta = paras
    chi2 = chi_square_SNe(log_kC1, O20, H0) + chi_square_QSO(log_kC1, O20, H0, gamma0, gamma1, beta0, beta1, delta)
    return -0.5 * chi2

def lnprior(paras):
    O20, log_kC1, H0, gamma0, gamma1, beta0, beta1, delta = paras
    if -10<log_kC1<0 and 0<O20<1 and 60<H0<80 and 0<gamma0<1 and -1<gamma1<1 and -15<beta0<15 and -10<beta1<10 and 0<delta<1:
        return 0.0
    return -np.inf

def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.3, -5, 70, 0.6, 0, 6, 1, 0.2])
    pos = initial + 1e-4 * np.random.randn(50, 8)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 5000, progress = True)
    flat_samples = sampler.get_chain(discard=500, flat=True)
    np.savetxt('./multimethods/output1.dat', flat_samples)

if __name__ == '__main__':
    mp.freeze_support()
    main()
