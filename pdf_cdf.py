import scipy
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import emcee
import seaborn as sns
import pandas as pd

import OHD.result
import SNe.result2
import BAO.result
import multimethods.methods2 as methods2
import multimethods.methods3 as methods3

import astropy.units as u
import astropy.constants as const
from cross_section import cross_section

def main():
    ## OHD
    initial = np.array([0.3, -5, 70])
    nll = lambda *args: -OHD.result.lnlike(*args)
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler_OHD = emcee.EnsembleSampler(nwalkers, ndim, OHD.result.lnprob, pool=pool)
        sampler_OHD.run_mcmc(pos, 2000, progress = True)

    flat_samples_OHD = sampler_OHD.get_chain(discard=200, flat=True)
    H0_OHD = np.median(flat_samples_OHD[:,2])
    H0_OHD_list = np.array([H0_OHD]*len(flat_samples_OHD))
    Mx_OHD = np.log10(cross_section(flat_samples_OHD[:,0], H0_OHD_list)) - flat_samples_OHD[:,1]

    ## SNe
    nll = lambda *args: -SNe.result2.lnlike(*args)
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler_SNe = emcee.EnsembleSampler(nwalkers, ndim, SNe.result2.lnprob, pool=pool)
        sampler_SNe.run_mcmc(pos, 2000, progress = True)

    flat_samples_SNe = sampler_SNe.get_chain(discard=200, flat=True)
    H0_SNe = np.median(flat_samples_SNe[:,2])
    H0_SNe_list = np.array([H0_SNe]*len(flat_samples_SNe))
    Mx_SNe = np.log10(cross_section(flat_samples_SNe[:,0], H0_SNe_list)) - flat_samples_SNe[:,1]

    ## OHD+SNe
    nll = lambda *args: -methods2.lnlike(*args)
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler_m2 = emcee.EnsembleSampler(nwalkers, ndim, methods2.lnprob, pool=pool)
        sampler_m2.run_mcmc(pos, 2000, progress = True)

    flat_samples_m2 = sampler_m2.get_chain(discard=200, flat=True)
    H0_m2 = np.median(flat_samples_m2[:,2])
    H0_m2_list = np.array([H0_m2]*len(flat_samples_m2))
    Mx_m2 = np.log10(cross_section(flat_samples_m2[:,0], H0_m2_list)) - flat_samples_m2[:,1]

    ## BAO
    nll = lambda *args: -BAO.result.lnlike(*args)
    initial = np.array([0.3, -5, 70, 100])
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 4)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler_BAO = emcee.EnsembleSampler(nwalkers, ndim, BAO.result.lnprob, pool=pool)
        sampler_BAO.run_mcmc(pos, 2000, progress = True)

    flat_samples_BAO = sampler_BAO.get_chain(discard=200, flat=True)
    H0_BAO = np.median(flat_samples_BAO[:,2])
    H0_BAO_list = np.array([H0_BAO]*len(flat_samples_BAO))
    Mx_BAO = np.log10(cross_section(flat_samples_BAO[:,0], H0_BAO_list)) - flat_samples_BAO[:,1]

    ## all combination
    nll = lambda *args: -methods3.lnlike(*args)
    initial = np.array([0.3, -5, 70, 100])
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 4)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, methods3.lnprob, pool=pool)
        sampler.run_mcmc(pos, 2000, progress = True)

    flat_samples = sampler.get_chain(discard=200, flat=True)
    H0 = np.median(flat_samples[:,2])
    H0_list = np.array([H0]*len(flat_samples))
    Mx = np.log10(cross_section(flat_samples[:,0], H0_list)) - flat_samples[:,1]

    # plot pdf
    pdf = pd.DataFrame({'OHD': flat_samples_OHD[:, 0], 'SN Ia': flat_samples_SNe[:, 0], 'OHD+SN Ia': flat_samples_m2[:, 0], 
                          'BAO': flat_samples_BAO[:, 0], 'OHD+SN Ia+BAO': flat_samples[:, 0]})
    plt.figure()
    sns.kdeplot(data=pdf, legend=True, bw_adjust=2, cut=0)
    plt.xlabel(r'$\Omega_{2,0}$')
    plt.savefig('./article//pictures/pdf_1.pdf')
    plt.show()
    # plot kc1 cdf
    cdf = pd.DataFrame({'OHD': flat_samples_OHD[:, 1], 'SN Ia': flat_samples_SNe[:, 1], 'OHD+SN Ia': flat_samples_m2[:, 1],
                          'BAO': flat_samples_BAO[:, 1], 'OHD+SN Ia+BAO': flat_samples[:, 1]})
    plt.figure()
    sns.ecdfplot(data=cdf, legend=True)
    plt.grid()
    plt.xlabel(r'$\log_{10}(\kappa C_1/Gyr^{-1})$')
    plt.savefig('./article/pictures/cdf_1.pdf')
    plt.show()
    # plot mx cdf
    cdf_ = pd.DataFrame({'OHD': Mx_OHD, 'SN Ia': Mx_SNe, 'OHD+SN Ia': Mx_m2, 'BAO': Mx_BAO, 'OHD+SN Ia+BAO': Mx})
    plt.figure()
    sns.ecdfplot(data=cdf_, legend=True)
    plt.grid()
    plt.xlabel(r'$\log_{10}(M_x/GeV)$')
    plt.savefig('./article/pictures/cdf_2.pdf')
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()