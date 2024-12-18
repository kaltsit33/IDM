import camb
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solution import solution
from solution import const_c
from cross_section import cross_section

file_path_TT = "./CMB/data/COM_PowerSpect_CMB-TT-full_R3.01.txt"
data_TT = np.loadtxt(file_path_TT, skiprows=1)
l_TT = data_TT[:,0]
Dl_TT = data_TT[:,1]
down_TT = data_TT[:,2]
up_TT = data_TT[:,3]

file_path_TE = "./CMB/data/COM_PowerSpect_CMB-TE-full_R3.01.txt"
data_TE = np.loadtxt(file_path_TE, skiprows=1)
l_TE = data_TE[:,0]
Dl_TE = data_TE[:,1]
down_TE = data_TE[:,2]
up_TE = data_TE[:,3]

file_path_EE = "./CMB/data/COM_PowerSpect_CMB-EE-full_R3.01.txt"
data_EE = np.loadtxt(file_path_EE, skiprows=1)
l_EE = data_EE[:,0]
Dl_EE = data_EE[:,1]
down_EE = data_EE[:,2]
up_EE = data_EE[:,3]

def caculation(log_kC1, O20, H0):
    z = solution(log_kC1, O20, H0)
    a = 1 / (1 + z.y[0,:])
    dota = -z.y[1,:] * a ** 2
    dtauda = 1 / (a * dota) * const_c
    omh2 = O20 * (H0 / 100) ** 2
    ombh2 = 0.0224

    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omh2-ombh2, omk=0, tau=0.054,  
                        As=2e-9, ns=0.965, halofit_version='mead', lmax=2600,
                        a_list=a, dtauda_list=dtauda)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['total']
    # totCL[:,0]=TT, totCL[:,1]=EE, totCL[:,3]=TE
    return totCL

def chi2(log_kC1, O20, H0):
    totCL = caculation(log_kC1, O20, H0)
    th_TT = totCL[int(l_TT[0]):int(l_TT[-1])+1,0]
    th_EE = totCL[int(l_EE[0]):int(l_EE[-1])+1,1]
    th_TE = totCL[int(l_TE[0]):int(l_TE[-1])+1,3]

    chi2_TT = np.sum((th_TT - Dl_TT)**2 / (down_TT**2 + up_TT**2))
    chi2_EE = np.sum((th_EE - Dl_EE)**2 / (down_EE**2 + up_EE**2))
    chi2_TE = np.sum((th_TE - Dl_TE)**2 / (down_TE**2 + up_TE**2))
    chi2 = chi2_TT + chi2_EE + chi2_TE
    return chi2

def plot(log_kC1, O20, H0):
    totCL = caculation(log_kC1, O20, H0)
    th_TT = totCL[int(l_TT[0]):int(l_TT[-1])+1,0]
    th_EE = totCL[int(l_EE[0]):int(l_EE[-1])+1,1]
    th_TE = totCL[int(l_TE[0]):int(l_TE[-1])+1,3]
    
    fig, ax = plt.subplots(2,2, figsize = (12,12))
    ax[0,0].plot(l_TT, th_TT, 'k', zorder=2)
    ax[0,0].errorbar(l_TT, Dl_TT, yerr=[down_TT, up_TT], fmt='ro', zorder=1)
    ax[0,0].set_title('TT')
    ax[0,1].plot(l_EE, th_EE, 'k', zorder=2)
    ax[0,1].errorbar(l_EE, Dl_EE, yerr=[down_EE, up_EE], fmt='bo', zorder=1)
    ax[0,1].set_title('EE')
    ax[1,0].plot(l_TE, th_TE, 'k', zorder=2)
    ax[1,0].errorbar(l_TE, Dl_TE, yerr=[down_TE, up_TE], fmt='go', zorder=1)
    ax[1,0].set_title('TE')
    ax[1,1].axis('off')
    ax[1,1].text(0, 0.5, 'chi2 = ' + str(chi2(log_kC1, O20, H0)), fontsize=15)
    plt.show()

import multiprocessing as mp
import emcee
import corner
import scipy

def r(z, paras):
    log_kC1, O20, H0, rsh = paras
    sol = solution(log_kC1, O20, H0)
    z0 = scipy.interpolate.interp1d(sol.t, sol.y[0,:], kind='cubic', fill_value='extrapolate')
    t0 = 1/H0
    t_z = scipy.optimize.root_scalar(lambda t: z0(t) - z, method='newton', x0=0)
    r_z = scipy.integrate.quad(lambda t: (1+z0(t)), t_z.root, t0)
    return r_z[0]

def chi_square(paras):
    # paras = [logkC1,O20,H0,rsh]
    z_star = 1090
    r_s_star = paras[3] / paras[2] * 1e2
    R = r(z_star, paras) * paras[2] * np.sqrt(paras[1])
    r_star = r(z_star, paras) * const_c
    l_A = np.pi * r_star / r_s_star
    R_data = [1.75,0.0089]
    l_data = [301.66,0.18]
    chi2_1 = (R - R_data[0])**2/(R_data[1]**2)
    chi2_2 = (l_A - l_data[0])**2/(l_data[1]**2)
    return chi2_1+chi2_2

def lnlike(paras):
    O20, log_kC1, H0, rsh = paras
    paras_ = [log_kC1, O20, H0, rsh]
    chi2 = chi_square(paras_)
    return -0.5 * chi2

def lnprior(paras):
    O20, log_kC1, H0, rsh = paras
    if 0 < O20 < 0.5 and -10 < log_kC1 < 0 and 50 < H0 < 80 and 50 < rsh < 150:
        return 0.0
    return -np.inf

def lnprob(paras):
    lp = lnprior(paras)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(paras)

def main():
    initial = np.array([0.3, -6.5, 67.5, 110]) # expected best values
    pos = initial + 1e-4 * np.random.randn(50, 4)
    nwalkers, ndim = pos.shape

    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(pos, 3000, progress = True)

    labels = [r'$\Omega_{2,0}$', r'$\log_{10}(\kappa C_1/$Gyr${}^{-1})$', '$H_0$[km/s/Mpc]', '$r_{*}h$']
    flat_samples = sampler.get_chain(discard=500, flat=True)
    figure1 = corner.corner(flat_samples, levels=(0.6826,0.9544), labels=labels, plot_datapoints=False, plot_density=False, fill_contours=True,
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14}, smooth=1, smooth1d=4, bins=50, hist_bin_factor=4, color='k')
    plt.tight_layout()
    plt.show()
    figure2 = corner.corner(flat_samples[:,0:2], levels=(0.6826,0.9544), labels=labels[0:2], plot_datapoints=False, plot_density=False, fill_contours=True,
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14}, smooth=1, smooth1d=4, bins=50, hist_bin_factor=4, color='k')
    plt.tight_layout()
    plt.show()

    H0 = np.median(flat_samples[:,2])
    H0_list = np.array([H0]*len(flat_samples))
    Mx = np.log10(cross_section(flat_samples[:,0], H0_list)) - flat_samples[:,1]
    combined_samples = np.vstack((flat_samples[:, 0], Mx)).T
    labels_ = [r'$\Omega_{2,0}$', r'$\log_{10}(M_x$/GeV)']
    figure = corner.corner(combined_samples, levels=(0.6826,0.9544), labels=labels_, plot_datapoints=False, plot_density=False, fill_contours=True,
                            title_fmt='.4f', show_titles=True, title_kwargs={"fontsize": 14}, smooth=1, smooth1d=4, bins=50, hist_bin_factor=4, color='k')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    mp.freeze_support()
    main()
