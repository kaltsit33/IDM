import numpy as np
import scipy
from decimal import Decimal, getcontext

import scipy.integrate
import scipy.interpolate
getcontext().prec = 100

import astropy.constants as const
import astropy.units as u

const_c =  const.c.value / 1000

distance_mpc = 1 * u.Mpc
distance_km = distance_mpc.to(u.km)
time_gyr = 1 * u.Gyr
time_s = time_gyr.to(u.s)
# Unit conversion (1/H0 to Gyr)
transfer = (distance_km / time_s).value

def solution(log_kC1, O20, n, H0, zmax):
    Omega_1_0 = 1 - O20
    Omega_2_0 = O20
    k = 10**log_kC1 * transfer

    def derivatives(a, y):
        H, I1, I2 = y

        xp_arg = np.clip(-2 * k * I1, -700, 700)

        dI1da = a**(n - 1) / H
        dI2da = (a**(-4)) / H * np.exp(xp_arg)
        
        N = Omega_2_0 * a**(-3) * np.exp(xp_arg)
        
        # D = 1 + (Omega_2_0 / Omega_1_0) * k * I2
        D = Decimal(1) + Decimal(Omega_2_0 / Omega_1_0) * Decimal(k) * Decimal(I2)
        D = float(D)
        
        dN_da = Omega_2_0 * (-3 * a**(-4) * np.exp(xp_arg) + 
                a**(-3) * np.exp(xp_arg) * (-2*k) * dI1da)
        
        dD_da = (Omega_2_0 / Omega_1_0) * k * dI2da
        
        # Term2_prime = (dN_da * D - N * dD_da) / D**2
        Term2_prime = Decimal(dN_da) / Decimal(D) - Decimal(N) / Decimal(D) * Decimal(dD_da) / Decimal(D)
        Term2_prime = float(Term2_prime)

        dHda = (H0**2 / (2 * H)) * (n * Omega_1_0 * a**(n-1) + Term2_prime)
        
        return [dHda, dI1da, dI2da]
    
    a_initial = 1.0
    y_initial = [H0, 0.0, 0.0]  # H(1)=H0, I1(1)=0, I2(1)=0

    a_end = 1 / (1 + zmax)
    a_span = (a_initial, a_end)

    result = scipy.integrate.solve_ivp(derivatives, a_span, y_initial, method='RK45', 
                        t_eval=np.linspace(a_initial, a_end, 10000))
    
    a_values = result.t
    H_values = result.y[0]

    return [a_values, H_values]

### OHD
file_path = "./OHD/OHD.dat"
pandata = np.loadtxt(file_path, skiprows=1)
z_hz = pandata[:, 0]
H_z = pandata[:, 1]
err_H = pandata[:, 2]

def chi_square_OHD(log_kC1, O20, n, H0):
    zmax = np.max(z_hz)
    a_values, H_values = solution(log_kC1, O20, n, H0, zmax)

    if np.sum(H_values<=0) > 0:
        return np.inf

    func = scipy.interpolate.interp1d(a_values, H_values, kind='cubic')

    H_th = []
    for z in z_hz:
        H_th.append(func(1 / (1 + z)))
    H_th = np.array(H_th)
    chi2 = np.sum((H_z - H_th)**2 / err_H**2)
    return chi2

### SNe Ia
file_path = "./SNe/Pantheon+ data/Pantheon+SH0ES.dat"
data = np.loadtxt(file_path, skiprows=1, usecols=(4, 6, 10))
z_cmb = data[:, 0]
z_hel = data[:, 1]
mu = data[:, 2]

file_path_cov = './SNe/Pantheon+ data/Pantheon+SH0ES_cov.dat'
cov = np.loadtxt(file_path_cov, skiprows=1)
cov_matrix = cov.reshape((1701, 1701))
cov_matrix_inv = np.linalg.inv(cov_matrix)

def chi_square_SNe(log_kC1, O20, n, H0):
    zmax = np.max(z_cmb)
    a_values, H_values = solution(log_kC1, O20, n, H0, zmax)

    if np.sum(H_values<=0) > 0:
        return np.inf

    func = scipy.interpolate.interp1d(a_values, H_values)
    dl = []
    for z in z_cmb:
        int_value = scipy.integrate.quad(lambda x: 1 / func(1 / (1 + x)), 0, z)[0]
        dl.append(const_c * int_value)
    dl = np.array(dl) * (1 + z_hel)
    muth = 5 * np.log10(dl) + 25
    delta_mu = muth - mu
    A = delta_mu @ cov_matrix_inv @ delta_mu.T
    B = np.sum(delta_mu @ cov_matrix_inv)
    C = np.sum(cov_matrix_inv)
    chi2 = A - B**2 / C + np.log(C / (2 * np.pi))
    return chi2

### BAO
data1 = np.loadtxt('./BAO/sdss.dat', skiprows=1)
data2 = np.loadtxt('./BAO/desi.dat', skiprows=1)
z_eff = np.concatenate((data1[:, 0], data2[:, 0]))
D_V_obs = np.concatenate((data1[:, 1], data2[:, 1]))
D_V_err = np.concatenate((data1[:, 2], data2[:, 2]))
D_M_obs = np.concatenate((data1[:, 3], data2[:, 3]))
D_M_err = np.concatenate((data1[:, 4], data2[:, 4]))
D_H_obs = np.concatenate((data1[:, 5], data2[:, 5]))
D_H_err = np.concatenate((data1[:, 6], data2[:, 6]))

def chi_square_BAO(log_kC1, O20, n, H0, rdh):
    zmax = np.max(z_eff)
    a_values, H_values = solution(log_kC1, O20, n, H0, zmax)

    if np.sum(H_values<=0) > 0:
        return np.inf

    func = scipy.interpolate.interp1d(a_values, H_values)

    def D_M(z):
        return const_c * scipy.integrate.quad(lambda x: 1 / func(1 / (1 + x)), 0, z)[0]
    def D_H(z):
        Hz = func(1 / (1 + z))
        return const_c / Hz
    def D_V(z):
        DM = D_M(z)
        DH = D_H(z)
        return (z * DM**2 * DH)**(1/3)

    rd = rdh / H0 * 100
    A, B, C = [0, 0, 0]
    for i in range(len(z_eff)):
        z = z_eff[i]
        if D_M_obs[i] != 0:
            A += (D_M_obs[i] - D_M(z) / rd)**2 / D_M_err[i]**2
        if D_H_obs[i] != 0:
            B += (D_H_obs[i] - D_H(z) / rd)**2 / D_H_err[i]**2
        if D_V_obs[i] != 0:
            C += (D_V_obs[i] - D_V(z) / rd)**2 / D_V_err[i]**2
    return A + B + C
    