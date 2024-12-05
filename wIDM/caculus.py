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
file_path = "./OHD/OHD.csv"
pandata = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=(0, 1, 2))
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
    chi2 = np.dot(delta_mu, np.dot(cov_matrix_inv, delta_mu))
    return chi2

### BAO
file_path_BAO = "./BAO/BAO.csv"
pandata_BAO = np.loadtxt(file_path_BAO, delimiter=',', skiprows=1, usecols=(3, 4, 5, 6, 7, 8, 9))
z_eff = pandata_BAO[:, 0]
D_M_obs = pandata_BAO[:, 1]
D_M_err = pandata_BAO[:, 2]
D_H_obs = pandata_BAO[:, 3]
D_H_err = pandata_BAO[:, 4]
D_V_obs = pandata_BAO[:, 5]
D_V_err = pandata_BAO[:, 6]

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
    