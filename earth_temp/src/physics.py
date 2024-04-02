import numpy as np
from numba import njit, prange
from numba import int64, float64
import scipy
from tqdm.auto import tqdm

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
def visible_intensity(alpha: float, n: int, T: float, h_max: float):
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
def temp_full_model(alpha_V, alpha_IR, n, T, h_max, surface_albedo=0, n_sweeps = 10)  -> float:
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
    Ts_in_V[-1] = 1    
    
    for s in range(n_sweeps):
        for i in range(2, n+1): # iterate backwards
            j = -i + 1 # density index
            
            Ts_in_V[-i]  = Ts_in_V[-i + 1] * np.exp(-sigma_V  * rhos[j] * dh)
            Ts_in_IR[-i] = (Ts_in_IR[-i + 1] + 1/2 * Es[-i + 1]) * np.exp(-sigma_IR * rhos[j] * dh)                
            
            Es[-i] = (1/2 * (Es[-i + 1] + Es[-i - 1]) + Ts_out[-i - 1] + Ts_in_IR[-i + 1]) \
                        * (1 - np.exp(-sigma_IR * rhos[j] * dh)) \
                        + Ts_in_V[-i + 1] * (1 - np.exp(-sigma_V * rhos[j] * dh))                   

        #try bottom layer as black body ground
        Es[0] = 1/2 * Es[1] + Ts_in_IR[1] + Ts_in_V[1] # Ts_in_
                
        Ts_out[0] = Es[0] # black body
        for i in range(1, n): # iterate forwards
            Ts_out[i]   =   (Ts_out[i - 1]   + 1/2 * Es[i - 1]) * np.exp(-sigma_IR * rhos[i] * dh)
        Ts_out[-1]      = Ts_out[-2]

    T = ((Es[0] * (1 - EARTH_ALBEDO) * SUN_FLUX) / SBC) ** (1/4)
    print("It. Outgoing:", Ts_out[-1])

    return T


@njit(parallel=True)
def temps_full_model_vary_sweeps(alpha_V, alpha_IR, n, T, h_max, n_sweeps_arr, surface_albedo=0):
    n_Ts = len(n_sweeps_arr)

    Ts = np.zeros(n_Ts)

    for i in prange(n_Ts):        
        Ts[i] = temp_full_model(alpha_V, alpha_IR, n, T, h_max, n_sweeps=n_sweeps_arr[i], surface_albedo=surface_albedo) - 273.15# ° C

    return Ts


@njit(parallel=True)
def temps_full_model_vary_alpha_IR(alpha_V, alphas_IR, n, T, h_max, surface_albedo=0)  -> float64[:]:
    n_Ts = len(alphas_IR)

    Ts = np.zeros(n_Ts)

    for i in prange(n_Ts):        
        Ts[i] = temp_full_model(alpha_V, alphas_IR[i], n, T, h_max, surface_albedo=surface_albedo) - 273.15# ° C

    return Ts

@njit(parallel=True)
def temps_full_model_vary_N(alpha_V, alpha_IR, ns, T, h_max, surface_albedo=0)  -> float64[:]:
    n_Ts = len(ns)

    Ts = np.zeros(n_Ts)

    for i in prange(n_Ts):         # TODO parallel
        Ts[i] = temp_full_model(alpha_V, alpha_IR, ns[i], T, h_max, surface_albedo=surface_albedo) -273.15 # ° C
    return Ts


@njit
def temp_full_model_matrix_approach(alpha_V, alpha_IR, n, T, h_max, surface_albedo=0)  -> float:
    # input alphas as attenuation coeffs
    sigma_V  = alpha_V  / RHO_0
    sigma_IR = alpha_IR / RHO_0 # TODO verify

    dh: float = h_max / n
    rhos = np.append(density_height(n, T, h_max), 0)

    # num variables: E, T_in_V, T_in_IR, T_out * n+1 = 4n + 4
    # indication: T_in_V, then T_in_IR, then T_out, then E
    coeffmat = np.zeros((4*n + 4, 4*n + 4), dtype=np.double)
    b = np.zeros(4*n + 4, dtype=np.double)
    
    # T_in_V
    # T[i] - exp * T[i+1] = 0
    coeffmat[n, n] = 1 # fix T_in == 1
    b[n] = 1
    for i in range(n): # last eqn set above
        j = i + 0 * (n + 1) # index of var along coeffmat diag
        coeffmat[j, j + 1] = -np.exp(-sigma_V  * rhos[i] * dh)
        coeffmat[j, j] = 1        
    
    # T_in_IR
    coeffmat[2*n+1, 2*n+1] = 1 # ingoing IR = 0
    for i in range(n):
        j = i + 1 * (n + 1) # T ir index
        k = i + 3 * (n + 1) # E index
        coeffmat[j, j + 1] = -np.exp(-sigma_IR  * rhos[i] * dh)
        coeffmat[j, k + 1] = -0.5 * np.exp(-sigma_IR  * rhos[i] * dh)
        coeffmat[j, j] = 1

    # T out
    coeffmat[2*n+2, 2*n+2] = 1 # lowest layer emits as black body
    coeffmat[2*n+2, 3*n+3] = -1
    coeffmat[3*n+2, 3*n+2] = 1 # outgoing IR = 1
    b[3*n+2] = 1
    for i in range(1, n):
        j = i + 2 * (n + 1) # T out index
        k = i + 3 * (n + 1) # E index
        coeffmat[j, j - 1] = -np.exp(-sigma_IR  * rhos[i] * dh)
        coeffmat[j, k - 1] = -0.5 * np.exp(-sigma_IR  * rhos[i] * dh)
        coeffmat[j, j] = 1
    
    # E
    coeffmat[3*n + 3, 3*n + 3] = -1 # lower layer is black body
    coeffmat[3*n + 3, 3*n + 4] = 0.5
    coeffmat[3*n + 3,   n + 2] = 1
    coeffmat[3*n + 3,       1] = 1 

    coeffmat[4*n + 3, 4*n + 3] = 1 # space has E = 0
    for i in range(1, n):
        j = i + 0 * (n + 1) # in v
        k = i + 1 * (n + 1) # in ir
        l = i + 2 * (n + 1) # out
        m = i + 3 * (n + 1) # E
        coeffmat[m, m] = -1
        coeffmat[m, m - 1] = 0.5 * (1 - np.exp(-sigma_IR  * rhos[i] * dh))
        coeffmat[m, m + 1] = 0.5 * (1 - np.exp(-sigma_IR  * rhos[i] * dh))
        coeffmat[m, l - 1] = (1 - np.exp(-sigma_IR  * rhos[i] * dh))
        coeffmat[m, k + 1] = (1 - np.exp(-sigma_IR  * rhos[i] * dh))
        coeffmat[m, j + 1] = (1 - np.exp(-sigma_V  * rhos[i] * dh))

    x = np.linalg.solve(coeffmat, b)
    T = ((x[3*n + 3] * (1 - EARTH_ALBEDO) * SUN_FLUX) / SBC) ** (1/4)
    print("Mat. ||Ax - b|| = ", np.linalg.norm(coeffmat @ x - b), "Outgoing:", x[3*n+2], "E:", x[3*n + 3])

    return T


def temps_full_model_vary_alpha_IR_mat(alpha_V, alphas_IR, n, T, h_max, surface_albedo=0)  -> float64[:]:
    n_Ts = len(alphas_IR)

    Ts = np.zeros(n_Ts)

    for i in range(n_Ts):        
        Ts[i] = temp_full_model_matrix_approach(alpha_V, alphas_IR[i], n, T, h_max, surface_albedo=surface_albedo) - 273.15 # ° C

    return Ts

def temps_full_model_vary_N_mat(alpha_V, alpha_IR, ns, T, h_max, surface_albedo=0)  -> float64[:]:
    n_Ts = len(ns)

    Ts = np.zeros(n_Ts)

    for i in range(n_Ts):
        Ts[i] = temp_full_model_matrix_approach(alpha_V, alpha_IR, ns[i], T, h_max, surface_albedo=surface_albedo) - 273.15 # ° C

    return Ts