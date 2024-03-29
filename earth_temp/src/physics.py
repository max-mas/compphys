import numpy as np
from numba import njit, prange
from numba import int64, float64

# constants
G = 6.674e-11 # N m^2 kg^-2
M_E = 5.972168e24 # kg
R_E = 6371e3 # m
K_B = 1.380649e-23
RHO_0 = 1.226 # kg m^-3, sea level density
P_0 = 101325 # Pa, sea level pressure
M_AIR = 4.809e-26 # kg, molecular mass of air
SBC = 5.670367e-8 # W m^-2 K^-4 stefan boltzmann constant
SUN_FLUX = 344 # W m^-2
EARTH_ALBEDO = 0.3

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
    rhos = M_AIR * Ps / (K_B * T)

    return rhos


# dI / dx = - sigma * rho * I
# => dI = - sigma * rho * I * dx
@njit
def visible_intensity(alpha, T, n, h_max):
    # input alpha as attenuation coeff
    sigma = alpha / RHO_0

    dh: float = h_max / n
    Is = np.zeros(n, dtype=np.double)
    rhos = density_height(n, T, h_max)

    Is[-1] = 1 # I at inf = h_max == I_0 = 1

    for i in range(2, n+1): # iterate backwards
        dI = sigma * rhos[-i] * dh 
        Is[-i] = Is[-i + 1] - dI
        if Is[-i] < 0:            
            Is[-i] = 0
            break # ensure I > = 0, attenuation may be strong enough to block everything)

    return Is


# get surface temp from visible light flux to surface
@njit
def temp_visible_light_absorption(alpha, n, T, h_max, surface_albedo=0):
    incoming_flux = SUN_FLUX * (1 - EARTH_ALBEDO)
    surface_flux = visible_intensity(alpha, n, T, h_max)[0] * incoming_flux * (1 - surface_albedo)
    temp = (surface_flux / SBC) ** (1/4)  # TODO wait, but we have to put T in to begin with?? >:(
    
    return temp


@njit(parallel=True)
def temps_visible_light_absorption(alphas, n, T, h_max, surface_albedo=0):
    Ts = np.zeros(len(alphas))

    for i in prange(len(alphas)):        
        Ts[i] = temp_visible_light_absorption(alphas[i], n, T, h_max, surface_albedo)

    return Ts

# TODO: then implement re-absorption etc?