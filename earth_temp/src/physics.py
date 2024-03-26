import numpy as np
from numba import njit
from numba import int64, float64

# constants
G = 6.674e-11 # N m^2 kg^-2
M_E = 5.972168e24 # kg
R_E = 6371e3 # m
K_B = 1.380649e-23
RHO_0 = 1.226 # kg m^-3, sea level density
P_0 = 101325 # Pa, sea level pressure
M_AIR = 4.809e-26 # kg, molecular mass of air

# TODO start with calculating density numerically (not explicitly w/ barometric formula)

@njit
# h: height over sea level (m)
# return: gravitational acceleration (m/s^2)
def grav(h: float) -> float:
    return G * M_E / (h + R_E)**2 # here g = |g|, so +

# derivation:
# PV = NkT => P = n * k * T = rho / m * k * T => rho = P * m / (k T)
# dP = - rho * g(h) * dh = - P m / (kT) * g(h) * dh
# n: number of cubes
# T: temp (K)
# h_max: where to end (m)
# return: P(h) (Pa)
@njit
def pressure_height(n, T, h_max) -> float64[:]: # type: ignore
    dh: float = h_max / n
    hs = np.linspace(0, h_max, n)
    Ps = np.zeros(n, dtype=np.double)
    Ps[0] = P_0

    for i, h in enumerate(hs, 1): # skip i == 0
        g = grav(h)
        dP = - Ps[i-1] * M_AIR / (K_B * T) * g * dh
        Ps[i] = Ps[i-1] + dP

    return Ps


# n: number of cubes
# T: temp (K)
# h_max: where to end (m)
# return: rho(h) (kg m^-3)
@njit
def density_height(n, T, h_max) -> float64[:]: # type: ignore
    Ps = pressure_height(n, T, h_max)
    rhos = Ps / (K_B * T) * M_AIR

    return rhos







# TODO next: simple radiation balance? not sure what's wanted here yet

# TODO: then implement re-absorption etc?