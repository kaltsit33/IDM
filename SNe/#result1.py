import numpy as np
from getdist import plots, MCSamples
import emcee
import corner
import scipy
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solution import solution
from solution import const_c
from cross_section import cross_section

H0 = 70.0
O20 = 0.3
log_kC1 = -5.0

# pantheon+
file_path = "./SNe/Pantheon+ data/Pantheon+SH0ES.dat"
pandata = np.loadtxt(file_path, skiprows=1, usecols=(2, 10))
z_hd = pandata[:, 0]
mu = pandata[:, 1]

file_path_cov = './SNe/Pantheon+ data/Pantheon+SH0ES_cov.dat'
cov = np.loadtxt(file_path_cov, skiprows=1)
cov_matrix = cov.reshape((1701, 1701))
cov_matrix_inv = np.linalg.inv(cov_matrix)

def chi_square(log_kC1, O20, H0):
    t0 = 1 / H0
    t_values = solution(log_kC1, O20, H0).t
    z_values = solution(log_kC1, O20, H0).y[0, :]
    dl_values = []

    for z_hz_value in z_hd:
        idx = np.searchsorted(z_values, z_hz_value)
        if idx >= len(z_values):  
            idx = len(z_values) - 1
        int_value = -np.trapezoid(z_values[:idx], t_values[:idx])
        dl_value = const_c * (1 + z_hz_value) * (t0 - t_values[idx] + int_value)
        dl_values.append(dl_value)

    dl = np.array(dl_values)
    muth = 5 * np.log10(dl) + 25
    delta_mu = muth - mu
    chi2 = delta_mu @ cov_matrix_inv @ delta_mu.T
    return chi2

def lnlike(paras):
    O20, log_kC1, H0 = paras
    chi2 = chi_square(log_kC1, O20, H0)
    return -0.5 * chi2

def lnprior(paras):
    O20, log_kC1, H0 = paras
    if 0 < O20 < 0.5 and -10 < log_kC1 < 0 and 60 < H0 < 80:
        return 0.0
    return -np.inf

def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.3, -5, 70])
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 2000, progress=True)

    labels = [r'\Omega_{2,0}', r'\log_{10}(\kappa C_1/Gyr{}^{-1})', 'H_0[km/s/Mpc]']
    names = ['O20', 'log_kC1', 'H0']
    flat_samples = sampler.get_chain(discard=200, flat=True)
    samples = MCSamples(samples=flat_samples, names=names, labels=labels, ranges={'log_kC1':(-10, None), 'H0':(60, 80)})
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, filled=True, contour_colors=['g'], title_limit=1)
    # g.export('./article/pictures/sne.pdf')
    g = plots.get_single_plotter(ratio=1)
    g.settings.axes_fontsize = 18
    g.settings.axes_labelsize = 24
    samples_2 = MCSamples(samples=flat_samples[:,0:2], names=names[0:2], labels=labels[0:2], ranges={'log_kC1':(-10, None)})
    g.plot_2d(samples_2, 'O20', 'log_kC1', filled=True, colors=['g'])
    # g.export('./article/pictures/sne_1.pdf')

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
    g.plot_2d(samples_, 'O20', 'Mx', filled=True, colors=['g'])
    # g.export('./article/pictures/sne_2.pdf')

if __name__ == '__main__':
    mp.freeze_support()
    main()