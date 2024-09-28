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

    flat_samples_OHD = sampler_OHD.get_chain(discard=100, flat=True)

    ## SNe
    nll = lambda *args: -SNe.result2.lnlike(*args)
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler_SNe = emcee.EnsembleSampler(nwalkers, ndim, SNe.result2.lnprob, pool=pool)
        sampler_SNe.run_mcmc(pos, 2000, progress = True)

    flat_samples_SNe = sampler_SNe.get_chain(discard=100, flat=True)

    ## OHD+SNe
    nll = lambda *args: -methods2.lnlike(*args)
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler_m2 = emcee.EnsembleSampler(nwalkers, ndim, methods2.lnprob, pool=pool)
        sampler_m2.run_mcmc(pos, 2000, progress = True)

    flat_samples_m2 = sampler_m2.get_chain(discard=100, flat=True)

    ## BAO
    nll = lambda *args: -BAO.result.lnlike(*args)
    initial = np.array([0.3, -5, 70, 100])
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 4)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler_BAO = emcee.EnsembleSampler(nwalkers, ndim, BAO.result.lnprob, pool=pool)
        sampler_BAO.run_mcmc(pos, 2000, progress = True)

    flat_samples_BAO = sampler_BAO.get_chain(discard=100, flat=True)

    ## all combination
    nll = lambda *args: -methods3.lnlike(*args)
    initial = np.array([0.3, -5, 70, 100])
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 4)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, methods3.lnprob, pool=pool)
        sampler.run_mcmc(pos, 2000, progress = True)

    flat_samples = sampler.get_chain(discard=100, flat=True)

    # plot pdf
    pdf = pd.DataFrame({'OHD': flat_samples_OHD[:, 0], 'SNe': flat_samples_SNe[:, 0], 'OHD+SNe': flat_samples_m2[:, 0], 
                          'BAO': flat_samples_BAO[:, 0], 'OHD+SNe+BAO': flat_samples[:, 0]})
    plt.figure()
    sns.kdeplot(data=pdf, legend=True)
    plt.xlabel(r'$\Omega_{2,0}$')
    plt.show()
    # plot cdf
    cdf = pd.DataFrame({'OHD': flat_samples_OHD[:, 1], 'SNe': flat_samples_SNe[:, 1], 'OHD+SNe': flat_samples_m2[:, 1],
                          'BAO': flat_samples_BAO[:, 1], 'OHD+SNe+BAO': flat_samples[:, 1]})
    plt.figure()
    sns.ecdfplot(data=cdf, legend=True)
    plt.grid()
    plt.xlabel(r'$\log_{10}\kappa C_1$')
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()