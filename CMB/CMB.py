# 导入包
import numpy as np
import matplotlib.pyplot as plt
import scipy
from time import time

from astropy.constants import c
const_c = c.to('km/s').value

def function(t, z, kC1, O10, H0):
    # z[0] = z(t), z[1] = z'(t), z[2] = z''(t)
    dz1 = z[1]
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
    # 求解区间
    tspan = (t0, 0)
    tn = np.linspace(t0, 0, 100000)
    # 从t0开始
    zt0 = [0, -H0]

    # t0给定初值
    z = scipy.integrate.solve_ivp(function, t_span=tspan, y0=zt0, t_eval=tn, method='RK45', args=(kC1, O10, H0))
    # z.y[0,:] = z(t), z.y[1,:] = z'(t)
    return z

class camb:
    def __init__(self, O20, log_kC1, H0):
        self.t0 = 1 / H0
        # dtau = dt / a
        self.t_list = np.array(solution(log_kC1, O20, H0).t)
        self.z_list = np.array(solution(log_kC1, O20, H0).y[0, :])
    # (flat) initial power spectrum
    # from planck 2018
    def Pk(self ,k):
        As = 2.092e-9
        ns = 0.9647
        n_run = 0.0011
        n_runrun = 0.009
        k0 = 0.05

        lnrat = np.log(k / k0)
        return As * np.exp(lnrat * (ns - 1 + lnrat * (n_run / 2 + lnrat * n_runrun / 6)))
    # tau0 - tau
    def delta_tau(self, t):
        idx = np.searchsorted(self.t_list, t)
        int_value = -np.trapz(self.z_list[:idx], self.t_list[:idx])
        r = self.t0 - t + int_value
        return r * const_c # Mpc
    
    def optimize(self):
        self.delta_tau_list = np.array(list(map(lambda t: self.delta_tau(t), self.t_list)))
    
    def Source_q(self):
        """ # calculate Source_q
        phidot = (1 / 2) * (adotoa * (-dgpi - 2 * k2 * phi) + dgq * k - diff_rhopi + k * sigma * (gpres + grho)) / k2]
        sigmadot = -adotoa*sigma - 1/2*dgpi/k + k*phi
        polter = pig/10+9/15*E(2)
        polterdot = (1/10)*pigdot + (3/5)*Edot(2)
        polterddot = -2/25*adotoa*dgq/(k*1) - 4/75*adotoa* k*sigma - 4/75*dgpi - 2/75*dgrho/1 - 3/ 50*k*octgdot*1 + (1/25)*k*qgdot - 1/5 \
                        *k*1*Edot(3) + (-1/10*pig + (7/10)* polter - 3/5*E(2))*dopacity + (-1/10*pigdot + (7/10)*polterdot - 3/5*Edot(2))*opacity

        ISW = 2 * phidot * exptau
        doppler = ((sigma + vb)*dvisibility + (sigmadot + vbdot)*visibility)/k
        monopole_source =  (-etak/(k*1) + 2*phi + clxg/4)*visibility
        quadrupole_source = (5 / 8)*(3*polter*ddvisibility + 6*polterdot*dvisibility + (k**2*polter + 3*polterddot)*visibility)/k**2
        Source_q[1] = ISW + doppler + monopole_source + quadrupole_source

        ang_dist = (tau0-taustart)
        Source_q[2] = visibility * polter * (15 / 8)/ (ang_dist ** 2 * k2) """
        Source_q1 = 100 # T mode
        Source_q2 = 1 # E mode
        return [Source_q1, Source_q2]
    
    def Delta_p_l_k(self, p, l, k):
        # tau from tmin to tmax
        xf = k * (self.delta_tau_list)
        J_l = scipy.special.spherical_jn(l, xf)
        int_value = -np.trapz(self.Source_q()[p-1] * J_l, self.t_list)
        return int_value
    # integrate dk/k * Delta_l_q ** 2 * P(k) 
    def iCl_scalar(self, l, method):
        k_max = np.inf
        if method == 'TT':
            def integrand(k, l):
                return (self.Delta_p_l_k(1, l, k) ** 2 * self.Pk(k) / k)
            return scipy.integrate.quad(integrand, 0, k_max, args=(l))[0]
        elif method == 'EE':
            def integrand(k, l):
                return (self.Delta_p_l_k(2, l, k) ** 2 * self.Pk(k) / k)
            return scipy.integrate.quad(integrand, 0, k_max, args=(l))[0]
        elif method == 'TE':
            def integrand(k, l):
                return (self.Delta_p_l_k(1, l, k) * self.Delta_p_l_k(2, l, k) * self.Pk(k) / k)
            return scipy.integrate.quad(integrand, 0, k_max, args=(l))[0]
        else:
            print('method not supported')
            return 0
    # output l(l+1)Cl/2pi
    def Cl_scalar(self, l, method):
        ctnorm = (l ** 2 - 1) * (l + 2) * l
        dbletmp = (l * (l + 1)) * 2
        if method == 'TT':
            return self.iCl_scalar(l, method) * dbletmp
        elif method == 'EE':
            return self.iCl_scalar(l, method) * ctnorm * dbletmp
        elif method == 'TE':
            return self.iCl_scalar(l, method) * np.sqrt(ctnorm) * dbletmp
        else:
            print('method not supported')
            return 0
    def ClTT(self, l):
        return self.Cl_scalar(l, 'TT')
    def ClEE(self, l):
        return self.Cl_scalar(l, 'EE')
    def ClTE(self, l):
        return self.Cl_scalar(l, 'TE')
        
import multiprocessing as mp
def main():
    # O20, log_kC1, H0
    O20 = 0.28
    log_kC1 = -3
    H0 = 70
    c = camb(O20, log_kC1, H0)
    c.optimize()
    l_list = np.arange(30, 2500)
    with mp.Pool() as pool:
        Cl_TT = pool.map(c.ClTT, l_list)
    plt.plot(l_list, Cl_TT, label='TT')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()