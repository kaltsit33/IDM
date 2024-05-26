# 其他程序的结果表明Mb不可限制，log_kC1非马尔可夫性，该程序限制O20与H0
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import scipy

const_c = 2.99792458e5
Mb = -18.9
# 取合适的地方进行约束
log_kC1 = -5

# 从txt文件中读取数据
file_path = "./SNe Ia/Pantheon.txt"
pandata = np.loadtxt(file_path, skiprows=3, usecols=(1, 4, 5))
# 提取第一列和第三列数据
z_hz = pandata[:, 0]
m = pandata[:, 1]
#mu=pandata[:,1]
#err_mu = pandata[:,2]
err_m = pandata[:, 2]

# 定义微分函数
def function(t, z, O20, H0):
    kC1 = 10**log_kC1
    O10 = 1 - O20
    # z[0] = z(t), z[1] = z'(t), z[2] = z''(t)
    dz1 = z[1]
    # 减少括号的使用,分为分子与分母
    up = H0**4 * kC1 * O10**2 * (z[0]**4+1) + 3 * H0**4 * O10**2 * z[0]**2 * (2 * kC1-3 * z[1]) \
        + H0**4 * O10**2 * z[0]**3 * (4 * kC1 - 3 * z[1]) - 3 * H0**4 * O10**2 * z[1] + 5 * H0**2 * O10 * z[1]**3\
            - kC1 * z[1]**4 + H0**2 * O10 * z[0] * (4 * H0**2 * kC1 * O10 - 9 * H0**2 * O10 * z[1] + 5 * z[1]**3)
    down = 2 * H0**2 * O10 * (1 + z[0])**2 * z[1]
    dz2 = up / down
    return [dz1, dz2]

# 解方程
def sov_func(O20, H0):
    t0 = 1/H0
    tspan = (t0, 0)
    tn = np.linspace(t0, 0, 100000)
    # 从t0开始
    zt0 = [0, -H0]

    # t0给定初值
    z = scipy.integrate.solve_ivp(function, t_span=tspan, y0=zt0, t_eval=tn, method='RK45', args={H0, O20})
    # z.y[0,:] = z(t), z.y[1,:] = z'(t)

    t_values = z.t
    z_values = z.y[0,:]

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

    return dl_values

# lnlike函数(对数似然函数)
def lnlike(paras):
    O20, H0 = paras
    # O20 = O20 % 1
    dl = np.array(sov_func(O20, H0))
    mth = Mb + 5 * np.log10(dl) + 25
    A = np.sum((m - mth)**2/err_m**2)
    B = np.sum((m - mth)/err_m**2)
    C = np.sum(1/err_m**2)
    chi2 = A - B**2/C + np.log(C/(2*np.pi))
    return -0.5 * chi2

# lnprior函数(先验概率函数)
def lnprior(paras):
    O20, H0 = paras
    if 0 < O20 < 0.5 and 50 < H0 < 100:
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
    initial = np.array([0.28, 70]) # expected best values
    soln = scipy.optimize.minimize(nll, initial)
    pos = soln.x + 1e-4 * np.random.randn(50, 2)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 500, progress = True)

    # 画图
    flat_samples = sampler.get_chain(discard=50, thin=10, flat=True)
    figure = corner.corner(flat_samples, bins=30, smooth=10, smooth1d=10, plot_datapoints=False, levels=(0.6826,0.9544), labels=[r'$\Omega_{2,0}$', '$H_0$'], 
                          color='royalblue', title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14})
    plt.show()

    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = [r'$\Omega_{2,0}$', '$H_0$']
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