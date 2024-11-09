# Import packages
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import scipy
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solution import solution

import astropy.units as u
import astropy.constants as const

G = const.G
c = const.c

section = 1e-23 * u.cm**3 / u.s

# Prior values
H0 = 70.0
O20 = 0.3
log_kC1 = -5.0

def cross_section(O20, H0):
    H0 = H0 * u.km / u.s / u.Mpc
    C1 = (1 - O20) * 3 * H0**2 / (8 * np.pi * G)
    cross = section * C1 * c**2
    cross = cross.to(u.GeV / u.Gyr)
    return cross.value
    
# Read data from csv file
file_path = "./OHD/OHD.csv"
pandata = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0, 1, 2))
z_hz = pandata[:, 0]
H_z = pandata[:, 1]
err_H = pandata[:, 2]

# Calculate chi-square
def chi_square(log_kC1, O20, H0):
    Z0 = solution(log_kC1, O20, H0).y[0,:]
    Z1 = solution(log_kC1, O20, H0).y[1,:]
    H_th = []
    for z0 in z_hz:
        # Find t1 corresponding to z1
        idx = np.searchsorted(Z0, z0)
        # If z_hz_value exceeds the range of z_values, use the last value
        if idx >= len(Z0):  
            idx = len(Z0) - 1
        z1 = Z1[idx]
        # Calculate H
        H_th.append(-1 / (1 + z0)*z1)

    H_th = np.array(H_th)
    chi2 = np.sum((H_z - H_th)**2 / err_H**2)
    return chi2

# lnlike function (log-likelihood function)
def lnlike(paras):
    O20, log_kC1, H0 = paras
    chi2 = chi_square(log_kC1, O20, H0)
    return -0.5 * chi2

# lnprior function (prior probability function)
def lnprior(paras):
    O20, log_kC1, H0 = paras
    if 0 < O20 < 0.7 and -10 < log_kC1 < 0 and 60 < H0 < 80:
        return 0.0
    return -np.inf

# lnprob function (log-posterior probability function)
def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    # Define MCMC parameters
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.3, -5, 70]) # expected best values
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    # Multithreaded MCMC
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 2000, progress = True)

    # MCMC result plot
    labels = [r'$\Omega_{2,0}$', r'$\log_{10}\kappa C_1$', '$H_0$']
    flat_samples = sampler.get_chain(discard=200, flat=True)
    figure = corner.corner(flat_samples, levels=(0.6826,0.9544), labels=labels, smooth=1,
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14}, bins=50)
    plt.show()

    H0 = np.median(flat_samples[:,2])
    H0_list = np.array([H0]*len(flat_samples))
    # Plot Mx vs O20 corner plot
    Mx = np.log10(cross_section(flat_samples[:,0], H0_list)) - flat_samples[:,1]
    combined_samples = np.vstack((flat_samples[:, 0], Mx)).T
    labels_ = [r'$\Omega_{2,0}$', r'$M_x$']
    figure = corner.corner(combined_samples, levels=(0.6826,0.9544), labels=labels_, smooth=1,
                           title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14}, bins=50)
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()