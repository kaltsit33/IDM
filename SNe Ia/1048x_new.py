import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import scipy

const_c=3e5

# 从txt文件中读取数据
file_path = "./Pantheon.txt"
pandata = np.loadtxt(file_path, skiprows=3, usecols=(1, 4, 5))
# 提取第一列和第三列数据
z_hz = pandata[:, 0]
m = pandata[:, 1]
#mu=pandata[:,1]
#err_mu = pandata[:,2]
err_m =pandata[:, 2]
a = 1/(1+z_hz)

# H0与t0作为已知初值
# H0 = 72
# t0 = 1/H0
# 先验值
# O2m = 0.3
# O1m = 1-O2m
# kC1 = 10**(-4.42)

# 把z重构为向量函数 z = [dz0,dz1,dz2]
def function(t, z, log_kC1, O2m, H0):
    kC1 = 10**log_kC1
    O1m = 1 - O2m
    # z[0] = z(t), z[1] = z'(t), z[2] = z''(t)
    dz1 = z[1]
    # 减少括号的使用,分为分子与分母
    up = H0**4*kC1*O1m**2*(z[0]**4+1)+3*H0**4*O1m**2*z[0]**2*(2*kC1-3*z[1])+H0**4*O1m**2*z[0]**3*(4*kC1-3*z[1])-3*H0**4*O1m**2*z[1]+5*H0**2*O1m*z[1]**3-kC1*z[1]**4+H0**2*O1m*z[0]*(4*H0**2*kC1*O1m-9*H0**2*O1m*z[1]+5*z[1]**3)
    down = 2*H0**2*O1m*(1+z[0])**2*z[1]
    dz2 = up/down
    return [dz1, dz2]

# 求解原方程对应的z(t)与z'(t) 
def sov_func(O2m,H0,log_kC1):
    t0 = 1/H0
    tspan = (t0,0)
    tn = np.linspace(t0,0,100000)
    # 从t0开始
    zt0 = [0,-H0]

    # t0给定初值
    z = scipy.integrate.solve_ivp(function,t_span=tspan,y0=zt0,t_eval=tn,args={log_kC1,O2m,H0},method='RK45')
    # z.y[0,:] = z(t), z.y[1,:] = z'(t)
    
    # 生成的解是np.ndarray类型,对解进行插值
    Z0=scipy.interpolate.interp1d(z.t,z.y[0,:],kind='nearest',bounds_error=False,fill_value='extrapolate')
    Z1=scipy.interpolate.interp1d(z.t,z.y[1,:],kind='nearest',bounds_error=False,fill_value='extrapolate')
   
    H_values = []
    z_values = np.linspace(0, 2, 2000)  
    for z1 in z_values:
        dZ = lambda t: Z0(t) - z1
        t1 = scipy.optimize.root(dZ, 0.003).x[0] # 确保解是标量
        H = -1/(1+Z0(t1))*Z1(t1)
        H_values.append(H)
        
    Dl_avg_values = []
    
    for z_hz_value in z_hz:
        # 找到 z_hz_value 在 z_values 中的位置
        idx = np.searchsorted(z_values, z_hz_value)
        
        # 如果 z_hz_value 超出了 z_values 的范围，使用最后一个值
        if idx >= len(z_values):  
            idx = len(z_values) - 1
        # 使用梯形规则计算 z0 和 z1 之间的面积
        H_values = np.array(H_values)
        # print(len(H_values[:idx]),len(z_values[1:idx+1]))
        area_z0_z1 = np.sum(((1/H_values[:idx] + 1/H_values[1:idx + 1]) * (z_values[1:idx + 1] - z_values[:idx]))/2)
    
        # 使用梯形规则计算 z0 和 z2 之间的面积
        area_z0_z2 = np.sum(((1/H_values[:idx - 1] + 1/H_values[1:idx]) * (z_values[1:idx] - z_values[:idx - 1]))/2) if idx > 0 else 0
    
        # 将这两个面积相加并除以 2，得到 z_hz 的梯形面积
        Dl_avg_value = const_c * (1 + z_hz_value) * ((area_z0_z1 + area_z0_z2) / 2)
        Dl_avg_values.append(Dl_avg_value)
    return Dl_avg_values

def mth(O2m, H0, log_kC1,Mb):
    Dl_avg_values = sov_func(O2m,H0,log_kC1)
    mth = Mb + 5 * np.log10(Dl_avg_values) + 25
    return mth

def lnlike(paras):
    O2m, H0, log_kC1, Mb = paras
    m_th = mth(O2m,H0,log_kC1,Mb)
#     print(np.sum(std))
    chi2 = np.sum((m_th - m)**2/err_m**2)
    return -0.5*chi2

# lnprior函数
def lnprior(paras):
    O2m, H0, log_kC1, Mb = paras
    if 0.0 < O2m < 1.0 and 50 < H0 < 100 and -10 < log_kC1 < -2 and -35< Mb < -5:
        return 0.0
    return -np.inf

# lnprob函数
def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

# 定义mcmc参量
nll = lambda *args: -lnlike(*args)
initial = np.array([0.3, 72, -5, -18.9]) # expected best values
soln = scipy.optimize.minimize(nll, initial)
pos = soln.x + 1e-4 * np.random.randn(50, 4)
nwalkers, ndim = pos.shape

# 多线程mcmc
import multiprocessing as mp

def main():
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 200, progress = True)

    # 画图
    samples = sampler.chain[:,20:].reshape((-1, ndim))
    fig = corner.corner(samples, bins=30, smooth=10, smooth1d=10, plot_datapoints=False,levels=(0.6826,0.9544), labels=[r'$\Omega_{2,m}$','$H_0$',r'$\log_{10}(\kappa C_1)$','$M_b$'], 
                          color='royalblue', title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14})
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()