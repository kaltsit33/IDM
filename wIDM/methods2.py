import numpy as np
import scipy
import matplotlib.pyplot as plt
import multiprocessing as mp
import emcee
import corner

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from caculus import chi_square_OHD
from caculus import chi_square_SNe

def lnlike(paras):
    O20, n, H0 = paras
    chi2 = chi_square_OHD(O20, n, H0) + chi_square_SNe(O20, n, H0)
    return -0.5 * chi2

def lnprior(paras):
    O20, n, H0 = paras
    if 0 < O20 < 1 and -10 < n < 10 and 50 < H0 < 100:
        return 0.0
    return -np.inf

def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.28, 0, 70]) # expected best values
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 2000, progress = True)

    labels = [r'$\Omega_{2,0}$', '$n$', '$H_0$']
    flat_samples = sampler.get_chain(discard=100, flat=True)
    figure = corner.corner(flat_samples, levels=(0.6826,0.9544), labels=labels, smooth=0.5, 
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14})
    plt.show()

    fig, axes = plt.subplots(3, figsize=(10, 9), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()