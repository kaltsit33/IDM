import numpy as np
import scipy

import astropy.constants as const

const_c =  const.c.value / 1000

H0 = 70
n = 1

def H(z, O20, n, H0):
    O10 = 1 - O20
    right = O10*(1 + z)**(-n) + O20*(1 + z)**3
    return np.sqrt(right) * H0

def H_ad(z, O20, n, H0):
    return 1/H(z, O20, n, H0)

### OHD
file_path = "./OHD/OHD.dat"
pandata = np.loadtxt(file_path, skiprows=1)
z_hz = pandata[:, 0]
H_z = pandata[:, 1]
err_H = pandata[:, 2]

def chi_square_OHD(O20, n, H0):
    H_th = []
    for z in z_hz:
        H_th.append(H(z, O20, n, H0))
    H_th = np.array(H_th)
    chi2 = np.sum((H_z - H_th)**2 / err_H**2)
    return chi2

### SNe Ia
file_path_SNe = "./SNe/Pantheon+ data/Pantheon+SH0ES.dat"
pandata_SNe = np.loadtxt(file_path_SNe, skiprows=1, usecols=(2, 10))
z_hd = pandata_SNe[:, 0]
mu = pandata_SNe[:, 1]

file_path_cov = './SNe/Pantheon+ data/Pantheon+SH0ES_cov.dat'
cov = np.loadtxt(file_path_cov, skiprows=1)
cov_matrix = cov.reshape((1701, 1701))
cov_matrix_inv = np.linalg.inv(cov_matrix)

def chi_square_SNe(O20, n, H0):
    dl = []
    for z in z_hd:
        int_value = scipy.integrate.quad(H_ad, 0, z, args=(O20, n, H0))[0]
        dl.append(const_c*(1 + z)*int_value)
    dl = np.array(dl)
    muth = 5 * np.log10(dl) + 25
    delta_mu = muth - mu
    A = delta_mu @ cov_matrix_inv @ delta_mu.T
    B = np.sum(delta_mu @ cov_matrix_inv)
    C = np.sum(cov_matrix_inv)
    chi2 = A - B**2 / C + np.log(C / (2 * np.pi))
    return chi2

### QSO
file_path_QSO = "./QSO/data/table3.dat"
data = np.loadtxt(file_path_QSO, skiprows=1, usecols=(3,4,5,6,7))
z_qso = data[:,0]
logFUV = data[:,1]
e_logFUV = data[:,2]
logFX = data[:,3]
e_logFX = data[:,4]

import astropy.units as u
transform = u.Mpc.to(u.m)

def logFX_z(O20, n, H0, gamma0, gamma1, beta0, beta1):
    dl = []
    for z in z_qso:
        int_value = scipy.integrate.quad(H_ad, 0, z, args=(O20, n, H0))[0]
        dl.append(const_c*(1 + z)*int_value)
    dl = np.array(dl) * transform

    beta = beta0 + beta1 * (1 + z_qso)
    gamma = gamma0 + gamma1 * (1 + z_qso)
    return beta+(gamma-1)*np.log10(4*np.pi)+gamma*logFUV+2*(gamma-1)*np.log10(dl)

def chi_square_QSO(O20, n, H0, gamma0, gamma1, beta0, beta1, delta):
    delta_fx = logFX_z(O20, n, H0, gamma0, gamma1, beta0, beta1) - logFX
    gamma = gamma0 + gamma1 * (1 + z_qso)
    sigma_2 = e_logFX**2 + gamma**2*e_logFUV**2 + delta**2
    chi2 = np.sum(delta_fx**2/sigma_2)
    extra = np.sum(np.log(2*np.pi*sigma_2))
    return chi2 + extra

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

class BAO:
    def __init__(self, O20, n, H0):
        self.O20 = O20
        self.n = n
        self.H0 = H0

    def D_M(self, z):
        int_value = scipy.integrate.quad(H_ad, 0, z, args=(self.O20, self.n, self.H0))[0]
        return int_value * const_c

    def D_H(self, z):
        Hz = H(z, self.O20, self.n, self.H0)
        return const_c / Hz

    def D_V(self, z):
        DM = self.D_M(z)
        DH = self.D_H(z)
        return (z * DM**2 * DH)**(1/3)

def chi_square_BAO(O20, n, H0, rdh):
    rd = rdh / H0 * 100
    theory = BAO(O20, n, H0)
    A, B, C = [0, 0, 0]
    for i in range(len(z_eff)):
        z = z_eff[i]
        if D_M_obs[i] != 0:
            A += (D_M_obs[i] - theory.D_M(z) / rd)**2 / D_M_err[i]**2
        if D_H_obs[i] != 0:
            B += (D_H_obs[i] - theory.D_H(z) / rd)**2 / D_H_err[i]**2
        if D_V_obs[i] != 0:
            C += (D_V_obs[i] - theory.D_V(z) / rd)**2 / D_V_err[i]**2
    return A + B + C
    