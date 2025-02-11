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

def lnlike(paras):
    O20, n, H0, gamma0, gamma1, beta0, beta1, delta = paras
    chi2 = chi_square_QSO(O20, n, H0, gamma0, gamma1, beta0, beta1, delta)
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

    labels = [r'\Omega_{2,0}', 'n', 'H_0[km/s/Mpc]', r'\gamma_0',r'\gamma_1',r'\beta_0',r'\beta_1',r'\delta']
    flat_samples = sampler.get_chain(discard=400, flat=True)
    samples = MCSamples(samples=flat_samples, names=labels, labels=labels)
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, filled=True, contour_colors=['k'], title_limit=1)
    plt.show()
    samples_2 = MCSamples(samples=flat_samples[:,0:2], names=labels[0:2], labels=labels[0:2])
    g.triangle_plot(samples_2, filled=True, contour_colors=['k'], title_limit=1)
    g.export('./article/pictures/qso_widm_1.pdf')
    plt.show()

    wIDM = -1 - flat_samples[:,1]/3
    combined_samples = np.vstack((flat_samples[:,0], wIDM)).T
    labels_ = [r'\Omega_{2,0}', 'w_{IDM}']
    samples_ = MCSamples(samples=combined_samples, names=labels_, labels=labels_)
    g.triangle_plot(samples_, filled=True, contour_colors=['k'], title_limit=1)
    g.export('./article/pictures/qso_widm_2.pdf')
    plt.show()

if __name__ == "__main__":
    mp.freeze_support()
    main()