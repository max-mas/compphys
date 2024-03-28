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
EARTH_ALBEDO = 0.33

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
    Ps[1] = Ps[0] + dh * -Ps[0] * M_AIR / (K_B * T) * grav(hs[0]) # Euler step
    Ps[2] = Ps[1] + dh + (-3/2 * Ps[1] * M_AIR / (K_B * T) * grav(hs[1]) \
                          +1/2 * Ps[0] * M_AIR / (K_B * T) * grav(hs[0]) ) # Two Step Adams–Bashforth

    for i, h in enumerate(hs, 3): # skip i == 0
        # three step Adams–Bashforth
        Ps[i] = Ps[i-1] + dh * (- 23/12 * Ps[i-1] * M_AIR / (K_B * T) * grav(hs[i-1]) \
                                + 16/12 * Ps[i-2] * M_AIR / (K_B * T) * grav(hs[i-2]) \
                                -  5/12 * Ps[i-3] * M_AIR / (K_B * T) * grav(hs[i-3]) )

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


# not necessary to solve deq, instead use solution
@njit
def visible_intensity(alpha, T, n, h_max):
    # input alpha as attenuation coeff
    sigma = alpha / RHO_0

    dh: float = h_max / n
    Is = np.zeros(n, dtype=np.double)
    rhos = density_height(n, T, h_max)

    Is[-1] = 1 # I at inf = h_max == I_0 = 1

    for i in range(2, n+1): # iterate backwards
        Is[-i] = Is[-i + 1] * np.exp(-sigma * rhos[-i] * dh)

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
@njit
def temp_full_model(alpha_V, alpha_IR, n, T, h_max, surface_albedo=0)  -> float:
    # input alphas as attenuation coeffs
    sigma_V  = alpha_V  / RHO_0
    sigma_IR = alpha_IR / RHO_0

    dh: float = h_max / n
    Es       = np.zeros(n+1, dtype=np.double) # +3 for init vals
    Ts_in_V  = np.zeros(n+1, dtype=np.double)
    Ts_in_IR = np.zeros(n+1, dtype=np.double)
    Ts_out   = np.zeros(n+1, dtype=np.double)
    rhos = density_height(n, T, h_max) # need to use different index for this later

    # init
    # Es[0], Es[1] can remain zero because there is no absorption w/o the atmosphere!
    Ts_in_V[-1] = 1
    Ts_in_V[-2] = 1
    Ts_in_V[-3] = 1
    Ts_in_IR[-1] = 1
    Ts_in_IR[-2] = 1
    Ts_in_IR[-3] = 1
    Ts_out[-1] = 1 # TODO is this correct? I think there should be as much going in as out
    Ts_out[-2] = 1
    Ts_out[-3] = 1


    for i in range(2, n+2): # iterate backwards
        j = -i + 1 # density index
        if i != n+1:
            Ts_in_V[-i]  =               Ts_in_V[-i + 1]           * np.exp(-sigma_V  * rhos[j] * dh)
            Ts_in_IR[-i] =   (Ts_in_IR[-i + 1] + 1/2 * Es[-i + 1]) * np.exp(-sigma_IR * rhos[j] * dh)
            Ts_out[-i]   =   (Ts_out[-i - 1]   + 1/2 * Es[-i - 1]) * np.exp(-sigma_IR * rhos[j] * dh)

            
            Es[-i]       =    (1/2 * (Es[-i + 1] + Es[-i - 1]) + Ts_out[-i - 1] + Ts_in_IR[-i + 1]) \
                            * (1 - np.exp(-sigma_IR * rhos[j] * dh)) \
                            + Ts_in_V[-i + 1] * (1 - np.exp(-sigma_V * rhos[j] * dh))
        else:
            Ts_in_V[-i]  =               Ts_in_V[-i + 1]           * np.exp(-sigma_V  * rhos[j] * dh)
            Ts_in_IR[-i] =   (Ts_in_IR[-i + 1] + 1/2 * Es[-i + 1]) * np.exp(-sigma_IR * rhos[j] * dh)
            Ts_out[-i]   =   0.0

            
            Es[-i]       =    (1/2 * Es[-i + 1] + Ts_in_IR[-i + 1]) * (1 - np.exp(-sigma_IR * rhos[j] * dh)) \
                            + Ts_in_V[-i + 1] * (1 - np.exp(-sigma_V * rhos[j] * dh))


    T = ((Es[0] * (1 - EARTH_ALBEDO) * SUN_FLUX) / SBC) ** (1/4)
    print(Es[0])

    return T

@njit(parallel=True)
def temps_full_model_vary_alpha_IR(alpha_V, alphas_IR, n, T, h_max, surface_albedo=0)  -> float64[:]:
    n_Ts = len(alphas_IR)

    Ts = np.zeros(n_Ts)

    for i in prange(n_Ts):        
        Ts[i] = temp_full_model(alpha_V, alphas_IR[i], n, T, h_max, surface_albedo=surface_albedo) - 273.15 # ° C

    return Ts