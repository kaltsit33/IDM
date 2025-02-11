import numpy as np
import scipy
import matplotlib.pyplot as plt
import multiprocessing as mp
import emcee
from getdist import plots, MCSamples

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from caculus import chi_square_SNe

def lnlike(paras):
    O20, n, H0 = paras
    chi2 = chi_square_SNe(O20, n, H0)
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
    initial = np.array([0.3, 0, 70]) # expected best values
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 2000, progress = True)

    labels = [r'\Omega_{2,0}', 'n', 'H_0[km/s/Mpc]']
    names = ['O20', 'n', 'H0']
    flat_samples = sampler.get_chain(discard=100, flat=True)
    samples = MCSamples(samples=flat_samples, names=names, labels=labels)
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, filled=True, contour_colors=['g'], title_limit=1)
    g.export('./article/pictures/sne_widm.pdf')
    samples_2 = MCSamples(samples=flat_samples[:,0:2], names=names[0:2], labels=labels[0:2])
    g = plots.get_single_plotter(ratio=1)
    g.settings.axes_fontsize = 18
    g.settings.axes_labelsize = 24
    g.plot_2d(samples_2, 'O20', 'n', filled=True, colors=['g'])
    g.export('./article/pictures/sne_widm_1.pdf')

    wIDM = -1 - flat_samples[:,1]/3
    combined_samples = np.vstack((flat_samples[:,0], wIDM)).T
    labels_ = [r'\Omega_{2,0}', 'w_{IDM}']
    names_ = ['O20', 'wIDM']
    samples_ = MCSamples(samples=combined_samples, names=names_, labels=labels_)
    g = plots.get_single_plotter(ratio=1)    
    g.settings.axes_fontsize = 18
    g.settings.axes_labelsize = 24
    g.plot_2d(samples_, 'O20', 'wIDM', filled=True, colors=['g'])
    g.export('./article/pictures/sne_widm_2.pdf')

if __name__ == "__main__":
    mp.freeze_support()
    main()