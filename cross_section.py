import numpy as np
import astropy.units as u
import astropy.constants as const

G = const.G
c = const.c

section = 1e-23 * u.cm**3 / u.s

def cross_section(O20, H0):
    H0 = H0 * u.km / u.s / u.Mpc
    C1 = (1 - O20) * 3 * H0**2 / (8 * np.pi * G)
    cross = section * C1 * c**2
    cross = cross.to(u.GeV / u.Gyr)
    return cross.value