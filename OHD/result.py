# 该程序实现OHD下的所有目标
# 导入包
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import scipy
from tqdm import tqdm
import multiprocessing as mp

# 常量
from astropy.constants import c
const_c = c.to('km/s').value

# 先验值
H0 = 70.0
O20 = 0.26
log_kC1 = -5.0

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
    z = scipy.integrate.solve_ivp(function, t_span=tspan, y0=zt0, t_eval=tn, 
                                  method='RK45', args=(kC1, O10, H0))
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
    if 0 < O20 < 0.5 and -5 < log_kC1 < 3 and 60 < H0 < 80:
        return 0.0
    return -np.inf

# lnprob函数(对数后验概率函数)
def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

# 主函数包括mcmc与格点法
def main():
    # 定义mcmc参量
    nll = lambda *args: -lnlike(*args)
    initial = np.array([0.26, -2, 70]) # expected best values
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
    figure = corner.corner(flat_samples, levels=(0.6826,0.9544), labels=labels,
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

    # 取出最佳H0
    H0 = np.percentile(flat_samples[:, 2], [50.0])[0]

     # 网格
    N = 100
    chi2 = np.zeros([N, N])
    log_kC1_list = np.linspace(3, -5, N)
    O20_list = np.linspace(0.0, 0.5, N)

    for i in tqdm(range(N), position=0, desc="O20", leave=False):
        for j in tqdm(range(N), position=1, desc="log_kC1", leave=False):
            log_kC1 = log_kC1_list[j]
            O20 = O20_list[i]
            # 较大值截断
            if chi_square(log_kC1, O20, H0) > 200:
                chi2[j][i] = 200
            else:
                chi2[j][i] = chi_square(log_kC1, O20, H0)

    # 3d plot
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    X, Y = np.meshgrid(O20_list, log_kC1_list)
    ax3.set_xlabel(labels[0])
    ax3.set_ylabel(labels[1])
    surf = ax3.plot_surface(X, Y, chi2, cmap='coolwarm')
    plt.colorbar(surf)
    plt.show()
    
    # 2d plot
    # 确定1sigma, 2sigma的边界
    O20_mcmc = np.percentile(flat_samples[:, 0], [2.28, 15.87, 50.0, 84.13, 97.72])
    chi2_1sigma = np.mean([chi_square(-5, O20_mcmc[1], H0), chi_square(-5, O20_mcmc[3], H0)])
    chi2_2sigma = np.mean([chi_square(-5, O20_mcmc[0], H0), chi_square(-5, O20_mcmc[4], H0)])
    # 画图
    plt.figure()
    contour = plt.contour(X, Y, chi2, [chi2_1sigma, chi2_2sigma], colors='k')
    plt.clabel(contour, inline=True, fontsize=8)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()