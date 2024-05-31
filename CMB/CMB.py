import numpy as np
import matplotlib.pyplot as plt
import scipy

from astropy.constants import c
const_c = c.to('km/s').value

# 定义微分函数
def function(t, z, kC1, O10, H0):
    # z[0] = z(t), z[1] = z'(t), z[2] = z''(t)
    dz1 = z[1]
    # 减少括号的使用,分为分子与分母
    up = H0 ** 4 * kC1 * O10 ** 2 * (z[0] ** 4+1) + 3 * H0 ** 4 * O10 ** 2 * z[0] ** 2 * (2 * kC1-3 * z[1]) \
        + H0 ** 4 * O10 ** 2 * z[0] ** 3 * (4 * kC1 - 3 * z[1]) - 3 * H0 ** 4 * O10 ** 2 * z[1] + 5 * H0 ** 2 * O10 * z[1] ** 3\
            - kC1 * z[1] ** 4 + H0 ** 2 * O10 * z[0] * (4 * H0 ** 2 * kC1 * O10 - 9 * H0 ** 2 * O10 * z[1] + 5 * z[1] ** 3)
    down = 2 * H0 ** 2 * O10 * (1 + z[0]) ** 2 * z[1]
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
    return [z.t, z.y[0, :], z.y[1, :]]

class CMB_TT:
    # 初始化并传递参数
    def __init__(self, log_kC1, O20, H0, As, ns, k0, xd):
        self.log_kC1 = log_kC1
        self.O20 = O20
        self.H0 = H0
        self.As = As
        self.ns = ns
        self.k0 = k0
        self.xd = xd
    def PR(self, k):
        def n(k):
            n1 = self.ns - 1
            n2 = 0.0011 * np.log(k / self.k0) / 2
            n3 = 0.009 * np.log(k / self.k0) ** 2 / 6
            return n1 + n2 + n3
        return self.As * (k / self.k0) ** n(k)
    def C_l(self, l):
        def j_l(x):
            return scipy.special.spherical_jn(l, x)
        def integrand(k):
            return self.PR(k) * j_l(k * self.xd) ** 2 * k ** 2
        return scipy.integrate.quad(integrand, 0, np.inf)[0] * 2 / np.pi
    
def r_t(log_kC1, O20, H0, z0):
    t0 = 1 / H0
    t_list = solution(log_kC1, O20, H0)[0]
    z_list = solution(log_kC1, O20, H0)[1]
    z = scipy.interpolate.interp1d(t_list, z_list, kind='cubic', fill_value='extrapolate', bounds_error=False)
    t = scipy.optimize.root_scalar(lambda t: z(t) - z0, method='newton', x0=0.00027).root
    r = scipy.integrate.quad(lambda t: (1 + z(t)), t, t0)[0]
    return r * const_c

import multiprocessing as mp    
def main():
    # 参数
    log_kC1 = -5
    O20 = 0.28
    H0 = 70
    As = 2.1
    ns = 0.96
    k0 = 0.05
    xd = r_t(log_kC1, O20, H0, 1090)
    # 实例化
    cmb = CMB_TT(log_kC1, O20, H0, As, ns, k0, xd)
    # 计算C_l
    l_list = np.linspace(30, 2500, 20)
    with mp.Pool() as pool:
        C_l = pool.map(cmb.C_l, l_list)
    
    D_l = l_list * (l_list + 1) * C_l / (2 * np.pi)
    plt.plot(l_list, D_l)
    plt.show()
    
if __name__ == '__main__':
    mp.freeze_support()
    main()