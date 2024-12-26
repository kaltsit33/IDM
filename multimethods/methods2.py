import numpy as np
import matplotlib.pyplot as plt
import emcee
from getdist import plots, MCSamples
import scipy
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cross_section import cross_section

# Prior values
H0 = 70.0
O20 = 0.3
log_kC1 = -5.0

### OHD
import OHD.result 
def chi_square_OHD(log_kC1, O20, H0):
    return OHD.result.chi_square(log_kC1, O20, H0)

### SNe Ia
import SNe.result2
def chi_square_SNe(log_kC1, O20, H0):
    return SNe.result2.chi_square(log_kC1, O20, H0)

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

    labels = [r'\Omega_{2,0}', r'\log_{10}(\kappa C_1/Gyr{}^{-1})', 'H_0[km/s/Mpc]']
    names = ['O20', 'log_kC1', 'H0']
    flat_samples = sampler.get_chain(discard=200, flat=True)
    samples = MCSamples(samples=flat_samples, names=names, labels=labels, ranges={'log_kC1':(-10, None)})
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, filled=True, contour_colors=['c'], title_limit=1)
    g.export('./article/pictures/ohd_sne.pdf')
    samples_2 = MCSamples(samples=flat_samples[:,0:2], names=names[0:2], labels=labels[0:2], ranges={'log_kC1':(-10, None)})
    g = plots.get_single_plotter(ratio=1)
    g.settings.axes_fontsize = 18
    g.settings.axes_labelsize = 24
    g.plot_2d(samples_2, 'O20', 'log_kC1', filled=True, colors=['c'])
    g.export('./article/pictures/ohd_sne_1.pdf')

    H0 = np.median(flat_samples[:,2])
    H0_list = np.array([H0]*len(flat_samples))
    # Plot Mx vs O20 corner plot
    Mx = np.log10(cross_section(flat_samples[:,0], H0_list)) - flat_samples[:,1]
    combined_samples = np.vstack((flat_samples[:, 0], Mx)).T
    labels_ = [r'\Omega_{2,0}', r'\log_{10}(M_x/GeV)']
    names_ = ['O20', 'Mx']
    samples_ = MCSamples(samples=combined_samples, names=names_, labels=labels_, ranges={'Mx':(None, -2)})
    g = plots.get_single_plotter(ratio=1)    
    g.settings.axes_fontsize = 18
    g.settings.axes_labelsize = 24
    g.plot_2d(samples_, 'O20', 'Mx', filled=True, colors=['c'])
    g.export('./article/pictures/ohd_sne_2.pdf')

if __name__ == '__main__':
    mp.freeze_support()
    main()
