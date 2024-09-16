# 导入包
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import scipy
import multiprocessing as mp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solution import solution
from solution import const_c

# 先验值
Mb = -19.3
H0 = 70.0
O20 = 0.28
log_kC1 = -5.0

# pantheon
file_path = "./SNe/Pantheon/Pantheon.txt"
pandata = np.loadtxt(file_path, skiprows=3, usecols=(1, 4, 5))
z_hz = pandata[:, 0]
m = pandata[:, 1]
err_m = pandata[:, 2]

# 计算卡方
def chi_square(log_kC1, O20, H0):
        t0 = 1 / H0
        t_values = solution(log_kC1, O20, H0).t
        z_values = solution(log_kC1, O20, H0).y[0, :]
        # 计算光度距离
        dl_values = []

        for z_hz_value in z_hz:
            # 找到 z_hz_value 在 z_values 中的位置
            idx = np.searchsorted(z_values, z_hz_value)
            # 如果 z_hz_value 超出了 z_values 的范围，使用最后一个值
            if idx >= len(z_values):  
                idx = len(z_values) - 1
            # 使用梯形规则计算 z0 和 z1 之间的面积
            int_value = -np.trapz(z_values[:idx], t_values[:idx])
            dl_value = const_c * (1 + z_hz_value) * (t0 - t_values[idx] + int_value)
            dl_values.append(dl_value)

        dl = np.array(dl_values)
        mth = Mb + 5 * np.log10(dl) + 25
        A = np.sum((m - mth)**2/err_m**2)
        B = np.sum((m - mth)/err_m**2)
        C = np.sum(1/err_m**2)
        chi2 = A - B**2/C + np.log(C/(2*np.pi))
        return chi2

# 这里的chi2已经去除了Mb的影响,故理论上不可限制H0,这在test中已有验证

# lnlike函数(对数似然函数)
def lnlike(paras):
    O20, log_kC1, H0 = paras
    chi2 = chi_square(log_kC1, O20, H0)
    return -0.5 * chi2

# lnprior函数(先验概率函数)
def lnprior(paras):
    O20, log_kC1, H0 = paras
    if 0 < O20 < 0.5 and -10 < log_kC1 < 0 and 60 < H0 < 80:
        return 0.0
    return -np.inf

# lnprob函数(对数后验概率函数)
def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    # 定义mcmc参量
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.28, -2, 70]) # expected best values
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    # 多线程mcmc
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 2000, progress = True)

    # mcmc结果图
    labels = [r'$\Omega_{2,0}$', r'$\log_{10}\kappa C_1$', '$H_0$']
    flat_samples = sampler.get_chain(discard=100, flat=True)
    # 采用默认格式
    figure = corner.corner(flat_samples, levels=(0.6826,0.9544), labels=labels, smooth=0.5,
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14})
    plt.show()

    # mcmc链图
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()