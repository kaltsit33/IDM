import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import scipy

const_c = 2.99792458e5

# 从csv文件中读取数据
file_path = "./OHD/OHD.csv"
pandata = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0, 1, 2))
z_hz = pandata[:, 0]
H_z = pandata[:, 1]
err_H = pandata[:, 2]

# 定义微分函数
def function(t, z, kC1, O10, H0):
    # z[0] = z(t), z[1] = z'(t), z[2] = z''(t)
    dz1 = z[1]
    # 减少括号的使用,分为分子与分母
    up = H0**4 * kC1 * O10**2 * (z[0]**4+1) + 3 * H0**4 * O10**2 * z[0]**2 * (2 * kC1-3 * z[1]) \
        + H0**4 * O10**2 * z[0]**3 * (4 * kC1 - 3 * z[1]) - 3 * H0**4 * O10**2 * z[1] + 5 * H0**2 * O10 * z[1]**3\
            - kC1 * z[1]**4 + H0**2 * O10 * z[0] * (4 * H0**2 * kC1 * O10 - 9 * H0**2 * O10 * z[1] + 5 * z[1]**3)
    down = 2 * H0**2 * O10 * (1 + z[0])**2 * z[1]
    dz2 = up / down
    return [dz1, dz2]

# 求解
def solution(log_kC1, O20, H0):
    kC1 = 10 ** log_kC1
    O10 = 1 - O20
    t0 = 1 / H0
    tspan = (t0, 0)
    tn = np.linspace(t0, 0, 100000)
    # 从t0开始
    zt0 = [0, -H0]

    # t0给定初值
    z = scipy.integrate.solve_ivp(function, t_span=tspan, y0=zt0, t_eval=tn, method='RK45', args=(kC1, O10, H0))
    # z.y[0,:] = z(t), z.y[1,:] = z'(t)
    return [z.y[0, :], z.y[1, :]]

# 计算卡方
def chi_square(log_kC1, O20, H0):
    Z0, Z1 = solution(log_kC1, O20, H0)
    H_th = []
    for z0 in z_hz:
        # 寻找z1对应的t1
        idx = np.searchsorted(Z0, z0)
        # 如果 z_hz_value 超出了 z_values 的范围，使用最后一个值
        if idx >= len(Z0):  
            idx = len(Z0) - 1
        z1 = Z1[idx]
        # 计算H
        H_th.append(-1 / (1 + z0) * z1)

    H_th = np.array(H_th)
    chi2 = np.sum((H_z - H_th)**2 / err_H**2)
    return chi2

# lnlike函数(对数似然函数)
def lnlike(paras):
    O20, log_kC1, H0 = paras
    chi2 = chi_square(log_kC1, O20, H0)
    return -0.5 * chi2

# lnprior函数(先验概率函数)
def lnprior(paras):
    O20, log_kC1, H0 = paras
    if 0 < O20 < 0.5 and -5 < log_kC1 < 3 and 50 < H0 < 100:
        return 0.0
    return -np.inf

# lnprob函数(对数后验概率函数)
def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

# 多线程mcmc
import multiprocessing as mp

def main():
    # 定义mcmc参量
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.26, -2, 70]) # expected best values
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 3)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 2000, progress = True)

    # 画图
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    figure = corner.corner(flat_samples, bins=30, smooth=10, smooth1d=10, plot_datapoints=False, levels=(0.6826,0.9544), labels=[r'$\Omega_{2,0}$', r'$\log_{10}k_{C1}$', r'$H_0$'], 
                          color='royalblue', title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14})
    plt.show()

    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r'$\Omega_{2,0}$', r'$\log_{10}k_{C1}$', r'$H_0$']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()

    tau = sampler.get_autocorr_time()
    print(tau)

if __name__ == '__main__':
    mp.freeze_support()
    main()