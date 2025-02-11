import numpy as np
import emcee
from getdist import plots, MCSamples
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cross_section import cross_section

H0 = 70.0
O20 = 0.3
log_kC1 = -5.0

import QSO.result2
def chi_square_QSO(log_kC1, O20, H0, gamma0, gamma1, beta0, beta1, delta):
    return QSO.result2.chi_square(log_kC1, O20, H0, gamma0, gamma1, beta0, beta1, delta)

import SNe.result1
def chi_square_SNe(log_kC1, O20, H0):
    return SNe.result1.chi_square(log_kC1, O20, H0)

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
    # soln = scipy.optimize.minimize(nll, initial)
    # pos = soln.x + 1e-4 * np.random.randn(50, 6)
    pos = initial + 1e-4 * np.random.randn(50, 8)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 2000, progress = True)

    labels = [r'\Omega_{2,0}',r'\log_{10}(\kappa C_1/Gyr{}^{-1})','H_0[km/s/Mpc]',r'\gamma_0',r'\gamma_1',r'\beta_0',r'\beta_1',r'\delta']
    names = ['O20','log_kC1','H0','gamma0','gamma1','beta0','beta1','delta']
    flat_samples = sampler.get_chain(discard=200, flat=True)
    samples = MCSamples(samples=flat_samples, names=names, labels=labels, ranges={'log_kC1':(-10, None), 'H0':(60,80)})
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, filled=True, contour_colors=['b'], title_limit=1)
    g.export('./article/pictures/sne_qso.pdf')
    samples_2 = MCSamples(samples=flat_samples[:,0:2], names=names[0:2], labels=labels[0:2], ranges={'log_kC1':(-10, None)})
    g = plots.get_single_plotter(ratio=1)
    g.settings.axes_fontsize = 18
    g.settings.axes_labelsize = 24
    g.plot_2d(samples_2, 'O20', 'log_kC1', filled=True, colors=['b'])
    g.export('./article/pictures/sne_qso_1.pdf')

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
    g.plot_2d(samples_, 'O20', 'Mx', filled=True, colors=['b'])
    g.export('./article/pictures/sne_qso_2.pdf')

if __name__ == '__main__':
    mp.freeze_support()
    main()
