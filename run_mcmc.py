import multiprocessing as mp
import scipy
import numpy as np
import emcee

def lnlike(paras):
    O20, log_kC1, H0 = paras
    chi2 = chi_square(log_kC1, O20, H0)
    return -0.5 * chi2

# lnprior函数(先验概率函数)
def lnprior(paras):
    O20, log_kC1, H0 = paras
    if 0 < O20 < 0.7 and -5 < log_kC1 < 3 and 60 < H0 < 80:
        return 0.0
    return -np.inf

# lnprob函数(对数后验概率函数)
def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def run_mcmc(initial, nwalkers=50, nsteps=2000):
    # 定义mcmc参量
    nll = lambda *args: -lnlike(*args)
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(nwalkers, len(initial))
    ndim = pos.shape[1]

    # 多线程mcmc
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
    
    return sampler