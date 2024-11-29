import camb
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solution import solution
from solution import const_c

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

def main():
    log_kC1_list = np.linspace(-8, -6, 20)
    O20_list = np.linspace(0.25, 0.35, 10)
    H0 = 67.5
    import concurrent.futures

    chi2_values = np.zeros((len(log_kC1_list), len(O20_list)))

    def compute_chi2(params):
        log_kC1, O20 = params
        return chi2(log_kC1, O20, H0)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(compute_chi2, [(log_kC1, O20) for log_kC1 in log_kC1_list for O20 in O20_list])

    for idx, result in enumerate(results):
        i = idx // len(O20_list)
        j = idx % len(O20_list)
        chi2_values[i, j] = result

    log_kC1_grid, O20_grid = np.meshgrid(log_kC1_list, O20_list, indexing='ij')
    plt.contourf(log_kC1_grid, O20_grid, chi2_values, levels=50, cmap='viridis')
    plt.colorbar(label='chi2')
    plt.xlabel('log_kC1')
    plt.ylabel('O20')
    plt.title('chi2 Contour Plot')
    plt.show()

if __name__ == "__main__":
    plot(-7.5, 0.3, 67.5)