import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from cross_section import cross_section

def main():
    upper_limit = {}
    lower_limit = {}

    flat_samples_OHD = np.loadtxt('./OHD/output.dat')
    H0_OHD = np.median(flat_samples_OHD[:,2])
    H0_OHD_list = np.array([H0_OHD]*len(flat_samples_OHD))
    Mx_OHD = np.log10(cross_section(flat_samples_OHD[:,0], H0_OHD_list)) - flat_samples_OHD[:,1]

    upper_limit['OHD'] = np.percentile(flat_samples_OHD[:,1], 97.7)
    lower_limit['OHD'] = np.percentile(Mx_OHD, 2.3)

    flat_samples_SNe = np.loadtxt('./SNe/output.dat')
    H0_SNe = np.median(flat_samples_SNe[:,2])
    H0_SNe_list = np.array([H0_SNe]*len(flat_samples_SNe))
    Mx_SNe = np.log10(cross_section(flat_samples_SNe[:,0], H0_SNe_list)) - flat_samples_SNe[:,1]

    upper_limit['SNe'] = np.percentile(flat_samples_SNe[:,1], 97.7)
    lower_limit['SNe'] = np.percentile(Mx_SNe, 2.3)

    flat_samples_m2 = np.loadtxt('./multimethods/output2.dat')
    H0_m2 = np.median(flat_samples_m2[:,2])
    H0_m2_list = np.array([H0_m2]*len(flat_samples_m2))
    Mx_m2 = np.log10(cross_section(flat_samples_m2[:,0], H0_m2_list)) - flat_samples_m2[:,1]

    upper_limit['OHD+SNe'] = np.percentile(flat_samples_m2[:,1], 97.7)
    lower_limit['OHD+SNe'] = np.percentile(Mx_m2, 2.3)

    flat_samples_m1 = np.loadtxt('./multimethods/output1.dat')
    H0_m1 = np.median(flat_samples_m1[:,2])
    H0_m1_list = np.array([H0_m1]*len(flat_samples_m1))
    Mx_m1 = np.log10(cross_section(flat_samples_m1[:,0], H0_m1_list)) - flat_samples_m1[:,1]

    upper_limit['SNe+QSO'] = np.percentile(flat_samples_m1[:,1], 97.7)
    lower_limit['SNe+QSO'] = np.percentile(Mx_m1, 2.3)

    flat_samples_BAO = np.loadtxt('./BAO/output.dat')
    H0_BAO = np.median(flat_samples_BAO[:,2])
    H0_BAO_list = np.array([H0_BAO]*len(flat_samples_BAO))
    Mx_BAO = np.log10(cross_section(flat_samples_BAO[:,0], H0_BAO_list)) - flat_samples_BAO[:,1]

    upper_limit['BAO'] = np.percentile(flat_samples_BAO[:,1], 97.7)
    lower_limit['BAO'] = np.percentile(Mx_BAO, 2.3)

    flat_samples = np.loadtxt('./multimethods/output3.dat')
    H0 = np.median(flat_samples[:,2])
    H0_list = np.array([H0]*len(flat_samples))
    Mx = np.log10(cross_section(flat_samples[:,0], H0_list)) - flat_samples[:,1]

    upper_limit['OHD+SNe+BAO'] = np.percentile(flat_samples[:,1], 97.7)
    lower_limit['OHD+SNe+BAO'] = np.percentile(Mx, 2.3)

    # plot pdf
    pdf = pd.DataFrame({'OHD': flat_samples_OHD[:, 0], 'SNe Ia': flat_samples_SNe[:, 0], 'SNe Ia+QSO': flat_samples_m1[:, 0],
                         'OHD+SNe Ia': flat_samples_m2[:, 0], 'BAO': flat_samples_BAO[:, 0], 'OHD+SNe Ia+BAO': flat_samples[:, 0]})

    fig, ax = plt.subplots()
    sns.kdeplot(data=pdf, fill=False, bw_adjust=4, cut=0, ax=ax)
    for line in ax.get_lines():
        x, y = line.get_data()
        y_norm = y / y.max()
        line.set_ydata(y_norm)
    ax.relim()
    ax.autoscale_view()
    ax.legend(labels=pdf.columns)
    ax.grid(False)
    ax.set_xlabel(r'$\Omega_{2,0}$')
    ax.set_ylabel(r'$P/P_{\max}$')
    plt.savefig('./article/pictures/pdf_1.pdf')
    plt.show()
    # plot kc1 cdf
    cdf = pd.DataFrame({'OHD': flat_samples_OHD[:, 1], 'SN Ia': flat_samples_SNe[:, 1], 'SN Ia+QSO': flat_samples_m1[:, 1],
                         'OHD+SN Ia': flat_samples_m2[:, 1], 'BAO': flat_samples_BAO[:, 1], 'OHD+SN Ia+BAO': flat_samples[:, 1]})
    plt.figure()
    sns.ecdfplot(data=cdf, legend=True)
    plt.grid()
    plt.xlabel(r'$\log_{10}(\kappa C_1/Gyr^{-1})$')
    plt.savefig('./article/pictures/cdf_1.pdf')
    plt.show()
    # plot mx cdf
    cdf_ = pd.DataFrame({'OHD': Mx_OHD, 'SN Ia': Mx_SNe, 'SN Ia+QSO': Mx_m1,
                          'OHD+SN Ia': Mx_m2, 'BAO': Mx_BAO, 'OHD+SN Ia+BAO': Mx})
    plt.figure()
    sns.ecdfplot(data=cdf_, legend=True)
    plt.grid()
    plt.xlabel(r'$\log_{10}(M_x/GeV)$')
    plt.savefig('./article/pictures/cdf_2.pdf')
    plt.show()

    print(upper_limit)
    print(lower_limit)

if __name__ == '__main__':
    main()