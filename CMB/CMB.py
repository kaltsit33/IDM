# 导入包
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
    up = H0**4 * kC1 * O10**2 * (z[0]**4+1) + 3 * H0**4 * O10**2 * z[0]**2 * (2 * kC1-3 * z[1]) \
        + H0**4 * O10**2 * z[0]**3 * (4 * kC1 - 3 * z[1]) - 3 * H0**4 * O10**2 * z[1] + 5 * H0**2 * O10 * z[1]**3\
            - kC1 * z[1]**4 + H0**2 * O10 * z[0] * (4 * H0**2 * kC1 * O10 - 9 * H0**2 * O10 * z[1] + 5 * z[1]**3)
    down = 2 * H0**2 * O10 * (1 + z[0])**2 * z[1]
    dz2 = up / down
    return [dz1, dz2]

# 求解
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
    return [z.t, z.y[0, :], z.y[1, :]]

# 约化尺度因子
def alpha(log_kC1, O20, H0):
    z = np.array(solution(log_kC1, O20, H0)[1])
    return 1 / (1 + z)

# 光子移动距离(单位c=1)
def r_t(log_kC1, O20, H0, t):
    t_list = np.array(solution(log_kC1, O20, H0)[0])
    idx = np.searchsorted(t_list, t)
    alpha_list = alpha(log_kC1, O20, H0)[:idx]
    r = -np.trapz(alpha_list, t_list[:idx])
    return r

# 计算CMB_TT时需要用到的各类函数,这里采用近似方法
class CMB_TT:
    # 初始化并传递参数
    def __init__ (self, log_kC1, O20, H0, others):
        self.log_kC1 = log_kC1
        self.O20 = O20
        self.H0 = H0
        # others=[T_0, z_L, t_L, r_L, alpha_L, dot_alpha_l, R_q^0, d_D, R_0]
        self.T0 = others[0]
        self.zL = others[1]
        self.tL = others[2]
        self.rL = others[3]
        self.aL = others[4]
        self.dal = others[5]
        self.Rq0 = others[6]
        self.dD = others[7]
        self.R0 = others[8]
        # 尺度因子
        self.RL = self.aL * self.R0
    # 3个近似函数
    def T(self, k):
        T1 = np.log(1 + (0.124 * k) ** 2) / (0.124 * k) ** 2
        T2 = 1 + (1.257 * k) ** 2 + (0.4452 * k) ** 4 + (0.2197 * k) ** 6
        T3 = 1 + (1.606 * k) ** 2 + (0.8568 * k) ** 4 + (0.3927 * k) ** 6
        return T1 * (T2 / T3) ** 0.5
    def S(self, k):
        S1 = 1 + (1.209 * k) ** 2 + (0.5116 * k) ** 4 + 5 ** 0.5 * (0.1657 * k) ** 6
        S2 = 1 + (0.9459 * k) ** 2 + (0.4249 * k) ** 4 + (0.1657 * k) ** 6
        return (S1 / S2) ** 2
    def Delta(self, k):
        Delta1 = (0.1585 * k) ** 2 + (0.9702 * k) ** 4 + (0.2460 * k) ** 6
        Delta2 = 1 + (1.180 * k) ** 2 + (1.540 * k) ** 4 + (0.9230 * k) ** 6 + (0.4197 * k) ** 8
        return (Delta1 / Delta2) ** 0.25
    # kappa函数
    def kappa(self, q):
        d_T = 0.035 / ((1 + self.zL) * self.H0) 
        return q * d_T / self.aL
    # 势函数
    def psi(self, q):
        psi1 = 3 * q ** 2 * self.tL * self.Rq0 * self.T(self.kappa(q))
        psi2 = 5 * self.aL ** 2
        return -psi1 / psi2
    # 2个微扰初始函数
    def dotB(self, q):
        return -2 * self.psi(q) / (q ** 2)
    def ddotB(self, q):
        # 差分法求导
        h = 1e-6
        return (self.dotB(q + h) - self.dotB(q - h)) / (2 * h)
    # 朗道阻尼修正
    def Gamma(self, q):
        return q ** 2 * self.dD ** 2 / self.aL ** 2
    # 光子函数中的积分
    def gamma_intvalue(self, q):
        t_list = np.array(solution(self.log_kC1, self.O20, self.H0)[0])
        idx = np.searchsorted(t_list, self.tL)
        def integrand(q):
            return q / (alpha(self.log_kC1, self.O20, self.H0)[idx:] * \
                        np.sqrt(3 * (1 + self.R0 * alpha(self.log_kC1, self.O20, self.H0)[idx:])))
        return np.trapz(integrand(q), t_list[idx:])
    # 2个光子函数
    def delta_gamma(self, q):
        delta_gamma1 = self.T(self.kappa(q)) * (1 + 3 * self.RL)
        delta_gamma2 = (1 + self.RL) ** -0.25 * np.exp(-self.Gamma(q)) * self.S(self.kappa(q))
        delta_gamma3 = np.cos(self.gamma_intvalue(q) + self.Delta(self.kappa(q)))
        return 3 * self.Rq0 / 5 * (delta_gamma1 - delta_gamma2 * delta_gamma3)
    def delta_u_gamma(self, q):
        delta_u_gamma1 = self.T(self.kappa(q)) * self.tL
        delta_u_gamma2 = self.aL / (np.sqrt(3) * (1 + self.R0 * self.aL) ** 0.75) * np.exp(-self.Gamma(q)) * self.S(self.kappa(q))
        delta_u_gamma3 = np.sin(self.gamma_intvalue(q) + self.Delta(self.kappa(q)))
        return 3 * self.Rq0 / 5 * (-delta_u_gamma1 + delta_u_gamma2 * delta_u_gamma3)
    # 2个时间独立函数
    def F(self, q):
        F1 = self.delta_gamma(q) / 3
        F2 = self.aL * self.ddotB(q) / 2
        F3 = self.aL * self.dal * self.dotB(q) / 2
        return F1 - F2 - F3
    def G(self, q):
        G1 = self.delta_u_gamma(q) / self.aL
        G2 = self.aL * self.dotB(q) / 2
        return -q * (G1 + G2)
    # 目标函数
    def C_l(self, l):
        def jl(l, q):
            return scipy.special.spherical_jn(l, q * self.rL)
        def jl_prime(l, q):
            return scipy.special.spherical_jn(l, q * self.rL, derivative=True)
        def integrand(l, q):
            return (jl(l, q) * self.F(q) + jl_prime(l, q) * self.G(q)) ** 2 * q ** 2
        def intvalue(l):
            return scipy.integrate.quad(integrand, 0, np.inf, args=(l))[0]
        return 16 * np.pi ** 2 * self.T0 ** 2 * intvalue(l)
    
import multiprocessing as mp
def main():
    # 参数
    log_kC1 = -5
    O20 = 0.2
    H0 = 70
    others = [2.725, 1100, 380000, 0.5, 0.5, 0.5, 1, 1, 1]
    # 实例化
    CMB = CMB_TT(log_kC1, O20, H0, others)
    # 计算
    l = np.linspace(2, 2500, 2500)
    with mp.Pool() as pool:
        C_l = pool.map(CMB.C_l, l)
    D_l = l * (l + 1) * C_l / (2 * np.pi)
    # 画图
    plt.plot(l, D_l)
    plt.xlabel('l')
    plt.ylabel('D_l')
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()