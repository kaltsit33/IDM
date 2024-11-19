import numpy as np
import scipy
import matplotlib.pyplot as plt
import multiprocessing as mp
import emcee
import corner

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from caculus import chi_square_BAO

def lnlike(paras):
    O20, n, H0, rdh = paras
    chi2 = chi_square_BAO(O20, n, H0, rdh)
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
    initial = np.array([0.28, 1, 70, 100]) # expected best values
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 4)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 5000, progress = True)

    labels = [r'$\Omega_{2,0}$', '$n$', '$H_0$[km/s/Mpc]', '$r_dh$[Mpc]']
    flat_samples = sampler.get_chain(discard=100, flat=True)
    figure = corner.corner(flat_samples, levels=(0.6826,0.9544), labels=labels, plot_datapoints=False, plot_density=False, fill_contours=True,
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14}, smooth=1, smooth1d=4, bins=50, hist_bin_factor=4, color='r')
    plt.tight_layout()
    plt.show()
    figure2 = corner.corner(flat_samples[:,0:2], levels=(0.6826,0.9544), labels=labels[0:2], plot_datapoints=False, plot_density=False, fill_contours=True,
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14}, smooth=1, smooth1d=4, bins=50, hist_bin_factor=4, color='r')
    plt.tight_layout()
    plt.savefig('./pictures/bao_widm_1.svg')
    plt.show()

    wIDM = -1 - flat_samples[:,1]/3
    combined_samples = np.vstack((flat_samples[:,0], wIDM)).T
    labels_ = [r'$\Omega_{2,0}$', '$w_{IDM}$']
    figure = corner.corner(combined_samples, levels=(0.6826,0.9544), labels=labels_, plot_datapoints=False, plot_density=False, fill_contours=True,
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14}, smooth=1, smooth1d=4, bins=50, hist_bin_factor=4, color='r')
    plt.tight_layout()
    plt.savefig('./pictures/bao_widm_2.svg')
    plt.show()