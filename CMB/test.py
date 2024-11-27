import camb
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solution import solution
from solution import const_c
const_c /= 1000

def main():
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

    z = solution(-6.5, 0.3, 67.3)
    a = 1 / (1 + z.y[0,:])
    dota = -z.y[1,:] * a ** 2
    dtauda = 1 / (a * dota) * const_c

    pars = camb.set_params(H0=67.3, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                        As=2e-9, ns=0.965, halofit_version='mead', lmax=2600,
                        a_list=a, dtauda_list=dtauda)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['total']
    # totCL[:,0]=TT, totCL[:,1]=EE, totCL[:,3]=TE
    th_TT = totCL[int(l_TT[0]):int(l_TT[-1])+1,0]
    th_EE = totCL[int(l_EE[0]):int(l_EE[-1])+1,1]
    th_TE = totCL[int(l_TE[0]):int(l_TE[-1])+1,3]

    chi2_TT = np.sum((th_TT - Dl_TT)**2 / (down_TT**2 + up_TT**2))
    chi2_EE = np.sum((th_EE - Dl_EE)**2 / (down_EE**2 + up_EE**2))
    chi2_TE = np.sum((th_TE - Dl_TE)**2 / (down_TE**2 + up_TE**2))
    chi2 = chi2_TT + chi2_EE + chi2_TE
    
    plt.figure()
    plt.plot(l_TT, Dl_TT, 'r-', label='TT')
    plt.plot(l_EE, Dl_EE, 'b-', label='EE')
    plt.plot(l_TE, Dl_TE, 'g-', label='TE')
    plt.plot(l_TT, th_TT, 'r--', label='TT_theory')
    plt.plot(l_EE, th_EE, 'b--', label='EE_theory')
    plt.plot(l_TE, th_TE, 'g--', label='TE_theory')
    plt.show()

if __name__ == "__main__":
    main()