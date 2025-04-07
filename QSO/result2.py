# use dm
import numpy as np
import matplotlib.pyplot as plt
import emcee
from getdist import plots, MCSamples
import scipy
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solution import solution
from solution import const_c
from cross_section import cross_section

import astropy.units as u
transform = u.Mpc.to(u.m)

H0 = 70.0
O20 = 0.3
log_kC1 = -5.0

file_path = "./QSO/data/table3.dat"
data = np.loadtxt(file_path, skiprows=1, usecols=(3,4,5,6,7))
z = data[:,0]
logFUV = data[:,1]
e_logFUV = data[:,2]
logFX = data[:,3]
e_logFX = data[:,4]

def logFX_z(log_kC1, O20, H0, gamma0, gamma1, beta0, beta1):
    t0 = 1 / H0
    t_values = solution(log_kC1, O20, H0).t
    z_values = solution(log_kC1, O20, H0).y[0, :]
    dl_values = []

    for z_value in z:
        idx = np.searchsorted(z_values, z_value)
        if idx >= len(z_values):  
            idx = len(z_values) - 1
        int_value = -np.trapezoid(z_values[:idx], t_values[:idx])
        dl_value = const_c * (1 + z_value) * (t0 - t_values[idx] + int_value)
        dl_values.append(dl_value)

    dl = np.array(dl_values) * transform
    beta = beta0 + beta1 * (1 + z)
    gamma = gamma0 + gamma1 * (1 + z)
    return beta+(gamma-1)*np.log10(4*np.pi)+gamma*logFUV+2*(gamma-1)*np.log10(dl)

def chi_square(log_kC1, O20, H0, gamma0, gamma1, beta0, beta1, delta):
    delta_fx = logFX_z(log_kC1, O20, H0, gamma0, gamma1, beta0, beta1) - logFX
    gamma = gamma0 + gamma1 * (1 + z)
    sigma_2 = e_logFX**2 + gamma**2*e_logFUV**2 + delta**2
    chi2 = np.sum(delta_fx**2/sigma_2)
    extra = np.sum(np.log(2*np.pi*sigma_2))
    return chi2 + extra

def lnlike(paras):
    O20, log_kC1, H0, gamma0, gamma1, beta0, beta1, delta = paras
    chi2 = chi_square(log_kC1, O20, H0, gamma0, gamma1, beta0, beta1, delta)
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
        sampler.run_mcmc(pos, 2000, progress=True)

    labels = [r'\Omega_{2,0}',r'\log_{10}(\kappa C_1/Gyr{}^{-1})','H_0[km/s/Mpc]',r'\gamma_0',r'\gamma_1',r'\beta_0',r'\beta_1',r'\delta']
    names = ['O20','log_kC1','H0','gamma0','gamma1','beta0','beta1','delta']
    flat_samples = sampler.get_chain(discard=200, flat=True)
    samples = MCSamples(samples=flat_samples, names=names, labels=labels, ranges={'log_kC1':(-10, None),'H0':(60, 80)})
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, filled=True, contour_colors=['k'], title_limit=1)
    g.export('./article/pictures/qso_fx.pdf')
    samples_2 = MCSamples(samples=flat_samples[:,0:2], names=names[0:2], labels=labels[0:2], ranges={'log_kC1':(-10, None)})
    g = plots.get_single_plotter(ratio=1)
    g.settings.axes_fontsize = 18
    g.settings.axes_labelsize = 24
    g.plot_2d(samples_2, 'O20', 'log_kC1', filled=True, colors=['k'])
    g.export('./article/pictures/qso_fx_1.pdf')

    H0 = np.median(flat_samples[:,2])
    H0_list = np.array([H0]*len(flat_samples))
    Mx = np.log10(cross_section(flat_samples[:,0], H0_list)) - flat_samples[:,1]
    combined_samples = np.vstack((flat_samples[:, 0], Mx)).T
    labels_ = [r'\Omega_{2,0}', r'\log_{10}(M_x/GeV)']
    names_ = ['O20', 'Mx']
    samples_ = MCSamples(samples=combined_samples, names=names_, labels=labels_, ranges={'Mx':(None, -2)})
    g = plots.get_single_plotter(ratio=1)    
    g.settings.axes_fontsize = 18
    g.settings.axes_labelsize = 24
    g.plot_2d(samples_, 'O20', 'Mx', filled=True, colors=['k'])
    g.export('./article/pictures/qso_fx_2.pdf')

if __name__ == '__main__':
    mp.freeze_support()
    main()