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
from solution import const_c
const_c /= 1000

# Prior values
H0 = 70.0
O20 = 0.28
log_kC1 = -5.0
rdh = 100

# Read data from csv file
file_path = "./BAO/BAO.csv"
pandata = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(3, 4, 5, 6, 7, 8, 9))
z_eff = pandata[:, 0]
D_M_obs = pandata[:, 1]
D_M_err = pandata[:, 2]
D_H_obs = pandata[:, 3]
D_H_err = pandata[:, 4]
D_V_obs = pandata[:, 5]
D_V_err = pandata[:, 6]

# Reduce the number of equation solutions
class BAO:
    def __init__(self, log_kC1, O20, H0, rd):
        self.log_kC1 = log_kC1
        self.O20 = O20
        self.H0 = H0
        self.rd = rd
        self.z_list = np.array(solution(log_kC1, O20, H0).y[0,:])
        self.zprime_list = np.array(solution(log_kC1, O20, H0).y[1,:])

    def D_M(self, z):
        idx = np.searchsorted(self.z_list, z)
        if idx >= len(self.z_list):
            idx = len(self.z_list) - 1
        intrange = self.z_list[:idx]
        integrand = -(1 + intrange) / self.zprime_list[:idx]
        intvalue = np.trapz(integrand, intrange)
        return intvalue * const_c

    def D_H(self, z):
        idx = np.searchsorted(self.z_list, z)
        if idx >= len(self.z_list):
            idx = len(self.z_list) - 1
        z1 = self.zprime_list[idx]
        Hz = -z1 / (1 + z)
        return const_c / Hz

    def D_V(self, z):
        DM = self.D_M(z)
        DH = self.D_H(z)
        return (z * DM**2 * DH)**(1/3)

def chi_square(log_kC1, O20, H0, rdh):
    rd = rdh / H0 * 100
    theory = BAO(log_kC1, O20, H0, rd)
    A, B, C = [0, 0, 0]
    for i in range(len(z_eff)):
        z = z_eff[i]
        if D_M_obs[i] != 0:
            A += (D_M_obs[i] - theory.D_M(z) / rd)**2 / D_M_err[i]**2
        if D_H_obs[i] != 0:
            B += (D_H_obs[i] - theory.D_H(z) / rd)**2 / D_H_err[i]**2
        if D_V_obs[i] != 0:
            C += (D_V_obs[i] - theory.D_V(z) / rd)**2 / D_V_err[i]**2
    return A + B + C

def lnlike(paras):
    O20, log_kC1, H0, rdh = paras
    chi2 = chi_square(log_kC1, O20, H0, rdh)
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
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.28, -5, 70, 100]) # expected best values
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 4)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 2000, progress = True)

    labels = [r'$\Omega_{2,0}$', r'$\log_{10}\kappa C_1$', '$H_0$', '$r_dh$']
    flat_samples = sampler.get_chain(discard=100, flat=True)

    figure = corner.corner(flat_samples, levels=(0.6826,0.9544), labels=labels, smooth=0.5, 
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14})
    plt.show()

    fig, axes = plt.subplots(4, figsize=(10, 9), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()