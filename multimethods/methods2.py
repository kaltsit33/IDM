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

### OHD
file_path_OHD = "./OHD/OHD.csv"
pandata_OHD = np.loadtxt(file_path_OHD, delimiter=',', skiprows=1, usecols=(0, 1, 2))
z_hz = pandata_OHD[:, 0]
H_z = pandata_OHD[:, 1]
err_H = pandata_OHD[:, 2]

class OHD:
    def __init__(self, log_kC1, O20, H0):
        self.log_kC1 = log_kC1
        self.O20 = O20
        self.H0 = H0
        self.Z0 = solution(log_kC1, O20, H0).y[0,:]
        self.Z1 = solution(log_kC1, O20, H0).y[1,:]

    def H_th(self, z):
        idx = np.searchsorted(self.Z0, z)
        if idx >= len(self.Z0):
            idx = len(self.Z0) - 1
        z1 = self.Z1[idx]
        return -1 / (1 + z) * z1
    
def chi_square_OHD(log_kC1, O20, H0):
    theory = OHD(log_kC1, O20, H0)
    chi2 = 0
    for i in range(len(z_hz)):
        z0 = z_hz[i]
        H_th = theory.H_th(z0)
        chi2 += (H_z[i] - H_th)**2 / err_H[i]**2
    return chi2

### SNe Ia
file_path_SNe = "./SNe/Pantheon+ data/Pantheon+SH0ES.dat"
pandata_SNe = np.loadtxt(file_path_SNe, skiprows=1, usecols=(2, 10, 11))
z_hd = pandata_SNe[:, 0]
mu = pandata_SNe[:, 1]
err_mu = pandata_SNe[:, 2]

class SNe:
    def __init__(self, log_kC1, O20, H0):
        self.log_kC1 = log_kC1
        self.O20 = O20
        self.H0 = H0
        self.t0 = 1 / H0
        self.t_values = solution(log_kC1, O20, H0).t
        self.z_values = solution(log_kC1, O20, H0).y[0, :]

    def DL(self, z):
        idx = np.searchsorted(self.z_values, z)
        if idx >= len(self.z_values):
            idx = len(self.z_values) - 1
        int_value = -np.trapz(self.z_values[:idx], self.t_values[:idx])
        dl_value = const_c * (1 + z) * (self.t0 - self.t_values[idx] + int_value)
        return dl_value
    
def chi_square_SNe(log_kC1, O20, H0):
    theory = SNe(log_kC1, O20, H0)
    dl_values = np.array([theory.DL(z) for z in z_hd])
    muth = 5 * np.log10(dl_values) + 25
    A = np.sum((mu - muth)**2 / err_mu**2)
    return A

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
        sampler.run_mcmc(pos, 2000, progress = True)

    labels = [r'$\Omega_{2,0}$', r'$\log_{10}\kappa C_1$', '$H_0$']
    flat_samples = sampler.get_chain(discard=200, flat=True)
    figure = corner.corner(flat_samples, levels=(0.6826,0.9544), labels=labels, smooth=1,
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14}, bins=50)
    plt.show()

    H0 = np.median(flat_samples[:,2])
    H0_list = np.array([H0]*len(flat_samples))
    Mx = np.log10(cross_section(flat_samples[:,0], H0_list)) - flat_samples[:,1]
    combined_samples = np.vstack((flat_samples[:, 0], Mx)).T
    labels_ = [r'$\Omega_{2,0}$', r'$M_x$']
    figure = corner.corner(combined_samples, levels=(0.6826,0.9544), labels=labels_, smooth=1,
                           title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14}, bins=50)
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()
