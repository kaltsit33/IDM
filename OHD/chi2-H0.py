import matplotlib.pyplot as plt
import numpy as np
import scipy
import multiprocessing as mp
from time import time

const_c = 2.99792458e5

# 从csv文件中读取数据
file_path = "./OHD/OHD.csv"
pandata = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0, 1, 2))
z_hz = pandata[:, 0]
H_z = pandata[:, 1]
err_H = pandata[:, 2]

N1 = 100
N2 = 100

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

def solution(log_kC1, O20, H0):
    kC1 = 10**log_kC1
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

def chi2_min(H0):
    start = time()
    chi2 = np.zeros([N1, N1])
    O20_list = np.linspace(0.1, 0.3, N1)
    log_kC1_list = np.linspace(0, 2, N1)
    for i in range(N1):
        for j in range(N1):
            log_kC1 = log_kC1_list[j]
            O20 = O20_list[i]
            chi2[j][i] = chi_square(log_kC1, O20, H0)
    # xx, yy = np.meshgrid(O20_list, log_kC1_list)
    # rb = scipy.interpolate.Rbf(xx, yy, chi2)
    # fun = lambda x: rb(x[0], x[1])
    # min = scipy.optimize.minimize(fun, [0.25, 1]).fun
    min = np.min(chi2)
    end = time()
    print(end - start)
    return min

def main():
    H0_list = np.linspace(60, 80, N2)
    with mp.Pool() as pool:
        chi2 = pool.map(chi2_min, H0_list)
        chi2 = np.array(chi2)

    plt.figure()
    plt.plot(H0_list, chi2)
    plt.xlabel(r'$H_0$')
    plt.ylabel(r'$\chi_{\min}^2$')
    plt.grid(linestyle='--', linewidth=0.5)
    # plt.savefig('Figure.png')
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()