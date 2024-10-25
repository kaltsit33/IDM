# Import packages
import numpy as np
import scipy

# Global fundamental constants
import astropy.constants as const
import astropy.units as u

const_c =  const.c.value

distance_gpc = 1 * u.Mpc
distance_km = distance_gpc.to(u.km)
time_gyr = 1 * u.Gyr
time_s = time_gyr.to(u.s)
# Unit conversion (1/H0 to Gyr)
transfer = (distance_km / time_s).value

# Reconstruct z as a vector function z = [dz0, dz1, dz2]
def function(t, z, kC1, O10, H0):
    # z[0] = z(t), z[1] = z'(t), z[2] = z''(t)
    dz1 = z[1]
    # Reduce the use of parentheses, separate into numerator and denominator
    numerator = (
        H0**4 * kC1 * O10**2 * (z[0]**4 + 1) +
        3 * H0**4 * O10**2 * z[0]**2 * (2 * kC1 - 3 * z[1]) +
        H0**4 * O10**2 * z[0]**3 * (4 * kC1 - 3 * z[1]) -
        3 * H0**4 * O10**2 * z[1] +
        5 * H0**2 * O10 * z[1]**3 -
        kC1 * z[1]**4 +
        H0**2 * O10 * z[0] * (4 * H0**2 * kC1 * O10 - 9 * H0**2 * O10 * z[1] + 5 * z[1]**3)
    )
    denominator = 2 * H0**2 * O10 * (1 + z[0])**2 * z[1]
    dz2 = numerator / denominator
    return [dz1, dz2]

# Solve the original equation for z(t) and z'(t)
def solution(log_kC1, O20, H0):
    # Unit conversion
    kC1 = 10**log_kC1 * transfer
    O10 = 1 - O20
    t0 = 1 / H0
    # Solution interval
    tspan = (t0, 0)
    tn = np.linspace(t0, 0, 100000)
    # Start from t0
    zt0 = [0, -H0]

    # Initial value given at t0
    z = scipy.integrate.solve_ivp(function, t_span=tspan, y0=zt0, t_eval=tn, method='RK45', args=(kC1, O10, H0))
    # z.y[0,:] = z(t), z.y[1,:] = z'(t)
    return z