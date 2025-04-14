import numpy as np
import emcee
import scipy
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solution import solution
from solution import const_c

H0 = 70.0
O20 = 0.3
log_kC1 = -5.0

# pantheon+
file_path = "./SNe/Pantheon+ data/Pantheon+SH0ES.dat"
data = np.loadtxt(file_path, skiprows=1, usecols=(4, 6, 10))
z_cmb = data[:, 0]
z_hel = data[:, 1]
mu = data[:, 2]

file_path_cov = './SNe/Pantheon+ data/Pantheon+SH0ES_cov.dat'
cov = np.loadtxt(file_path_cov, skiprows=1)
cov_matrix = cov.reshape((1701, 1701))
cov_matrix_inv = np.linalg.inv(cov_matrix)

def chi_square(log_kC1, O20, H0):
    t0 = 1 / H0
    t_values = solution(log_kC1, O20, H0).t
    z_values = solution(log_kC1, O20, H0).y[0, :]
    dl_values = []

    for z_value in z_cmb:
        idx = np.searchsorted(z_values, z_value)
        if idx >= len(z_values):  
            idx = len(z_values) - 1
        int_value = -np.trapezoid(z_values[:idx], t_values[:idx])
        dl_value = const_c * (t0 - t_values[idx] + int_value)
        dl_values.append(dl_value)

    dl = np.array(dl_values * (1 + z_hel))
    muth = 5 * np.log10(dl) + 25
    delta_mu = muth - mu
    A = delta_mu @ cov_matrix_inv @ delta_mu.T
    B = np.sum(delta_mu @ cov_matrix_inv)
    C = np.sum(cov_matrix_inv)
    chi2 = A - B**2 / C + np.log(C / (2 * np.pi))
    return chi2

def lnlike(paras):
    O20, log_kC1, H0 = paras
    chi2 = chi_square(log_kC1, O20, H0)
    return -0.5 * chi2

def lnprior(paras):
    O20, log_kC1, H0 = paras
    if 0 < O20 < 1.0 and -10 < log_kC1 < 0 and 60 < H0 < 80:
        return 0.0
    return -np.inf

def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.3, -5, 70])
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 5000, progress=True)
    flat_samples = sampler.get_chain(discard=500, flat=True)
    np.savetxt('./SNe/output.dat', flat_samples)

if __name__ == '__main__':
    mp.freeze_support()
    main()
