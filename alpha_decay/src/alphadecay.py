import numpy as np
import scipy as scp
from numba import njit, int64, float64, complex64
from numba.types import UniTuple, DictType
mass_dict_key_type = UniTuple(int64, 2)
mass_dict_type = DictType(mass_dict_key_type, float64)
from numba.experimental import jitclass
from numba.typed import Dict

# unit conversion constants:
hc = 197.3269631 # hbar * c (MeV * fm)
fs = 7.2973525693e-3 # fine structure constant (dimensionless)
M_alpha = 3727.379 #  alpha particle mass (MeV)

@jitclass
class Alphadecay:    
    # typing of member variables
    A: int
    Z: int
    R: float
    V0: float
    E_kin: float
    discr_steps: int
    coulomb_rng: float    
    mass_dict: mass_dict_type # type: ignore
    solved: bool
    x: complex64[:] # type: ignore
    psi: complex64[:] # type: ignore
    density: float64[:] # type: ignore

    # exact coulomb
    # r: distance (fm)
    # return: energy (MeV)
    def coulomb(self, r) -> float:
        Z1 = self.Z - 2
        Z2 = 2
        return fs * Z1 * Z2 * hc / r

    # E kin from mass defect in the frame where the daughter nucleus is stationary
    # return: energy (MeV)
    def E_kin_alpha(self):
        M_1 = self.mass_dict[(self.Z, self.A)]
        M_2 = self.mass_dict[(self.Z-2, self.A-4)]
        E_defect = M_1 - M_2 - M_alpha

        v_rel_squared = 2 * E_defect * (M_2 + M_alpha) / (M_2 * M_alpha) # (c)
        E_alpha = 0.5 * v_rel_squared * M_alpha 

        return E_alpha


    # TODO function that generates the discretised potential
    def piecewise_constant_potential(self):
        V = np.empty(self.discr_steps + 2, dtype=np.double)
        dx = (self.coulomb_rng - self.R) / self.discr_steps # TODO does R factor in here?

        V[0] = self.V0
        for i in range(1, self.discr_steps + 2):
            x = self.R + (i-1) * dx
            V[i] = self.coulomb(x)
        
        return V
    
    # TODO k / kappa
    def k(self, V):
        return np.sqrt(2 * M_alpha * np.abs(self.E_kin - V)) / hc
    
    def kappa(self, V):
        return np.sqrt(2 * M_alpha * np.abs(V - self.E_kin)) / hc


    # TODO function that makes the matrix for the LSE
    def coeff_mat(self):
        xs = np.ones(self.discr_steps + 1, dtype=np.double) * self.R 
        dx = (self.coulomb_rng - self.R) / self.discr_steps
        for i in range(self.discr_steps + 1):
            xs[i] += i * dx

        V = self.piecewise_constant_potential()
        
        A = np.zeros((2 * self.discr_steps + 3, 2 * self.discr_steps + 3), dtype=np.complex64)
        A[0, 0] = 1
        x = xs[0]
        V1 = V[0]
        V2 = V[1]
        A[1, 0] =  np.exp( 1j * self.k(V1) * x)
        A[1, 1] =  np.exp(-1j * self.k(V1) * x)
        A[1, 2] = -np.exp( self.kappa(V2) * x)
        A[1, 3] = -np.exp(-self.kappa(V2) * x)
        A[2, 0] =  1j * self.k(V1) * np.exp( 1j * self.k(V1) * x)
        A[2, 1] = -1j * self.k(V1) * np.exp(-1j * self.k(V1) * x) 
        A[2, 2] = -     self.kappa(V2) * np.exp( self.kappa(V2) * x) 
        A[2, 3] =       self.kappa(V2) * np.exp(-self.kappa(V2) * x) 

        for i in range(3, 2*self.discr_steps + 1):
            j = (i-1) // 2 
            V1 = V[j]
            V2 = V[j + 1]
            x = xs[j]
                     
            if i % 2 == 1:
                A[i, 2*j    ] =  np.exp( self.kappa(V1) * x)
                A[i, 2*j + 1] =  np.exp(-self.kappa(V1) * x)
                A[i, 2*j + 2] = -np.exp( self.kappa(V2) * x)
                A[i, 2*j + 3] = -np.exp(-self.kappa(V2) * x)
            else:
                A[i, 2*j]       =  self.kappa(V1) * np.exp( self.kappa(V1) * x)
                A[i, 2*j + 1]   = -self.kappa(V1) * np.exp(-self.kappa(V1) * x) 
                A[i, 2*j + 2]   = -self.kappa(V2) * np.exp( self.kappa(V2) * x) 
                A[i, 2*j + 3]   =  self.kappa(V2) * np.exp(-self.kappa(V2) * x) 
            
        x = xs[-1]
        V1 = V[-2]
        V2 = V[-1]
        A[-2, -3] =  np.exp( self.kappa(V1) * x)
        A[-2, -2] =  np.exp(-self.kappa(V1) * x)
        A[-2, -1] = -np.exp( 1j * self.k(V2) * x)
        A[-1, -3] =   self.kappa(V1) * np.exp( self.kappa(V1) * x) 
        A[-1, -2] = - self.kappa(V1) * np.exp(-self.kappa(V1) * x)
        A[-1, -1] = -1j * self.k(V2) * np.exp( 1j * self.k(V2) * x)

        return A

    # TODO function that solves the LSE and returns the coefficients
    def solve(self):
        A = self.coeff_mat()
        b = np.zeros(2 * self.discr_steps + 3, dtype=np.complex64)
        b[0] = 1
        self.x = np.linalg.solve(A, b)
        self.solved = True

        return self.x

    # TODO function that returns the transmission coefficient
    def get_transm_coeff(self):
        if not self.solved:
            self.solve()

        V = self.piecewise_constant_potential()
        k1 = self.k(V[0])
        k2 = self.k(V[-1])

        T = np.abs(self.x[-1]) ** 2 * k2 / k1
        return T
    
    # TODO function that gets the half life from T
    def get_half_life(self): # seconds
        v = np.sqrt(2 * self.E_kin / M_alpha) * 2.998e23  # (fm/s)
        tau = 2 * self.R / (v * self.get_transm_coeff())
        t_12 = tau * np.log(2)
        
        return t_12 #/ (60 * 24 * 365) #years

    # TODO function that builds the wave function from the coefficients
    # x range: 0 to coulomb range + 10 fm
    # n: number of evaluation points
    def calculate_psi(self, n=1000):
        if not self.solved:
            self.solve()
        
        xs = np.linspace(0.0, self.coulomb_rng + 10, n)
        dx = (self.coulomb_rng - self.R) / self.discr_steps
        V = self.piecewise_constant_potential()
        psi = np.zeros(n, dtype=np.complex64)

        for i, x in enumerate(xs):
            if x <= self.R:                
                psi[i] = self.x[0] * np.exp( 1j * self.k(V[0]) * x) + self.x[1] * np.exp( -1j * self.k(V[0]) * x)
            elif x >= self.coulomb_rng:
                psi[i] = self.x[-1] * np.exp( 1j * self.k(V[-1]) * x)
            else:
                j = int((x - self.R) / dx) + 1          
                psi[i] = self.x[2*j] * np.exp(self.kappa(V[j]) * x) \
                    + self.x[2*j + 1] * np.exp(-self.kappa(V[j]) * x)
        
        self.psi = psi
        return psi
            
    
    def calculate_density(self, n):
        if not self.solved:
            self.solve()

        self.calculate_psi(n)
        
        self.density = (self.psi.real ** 2 + self.psi.imag ** 2).astype(np.float64)
        
        return self.density

         

    # A: Atomic mass number
    # Z: Charge number
    # R: Core radius (fm)
    # V0: Depth of core well wrt. alpha particle energy (MeV)
    # discr_steps: Number of discrete steps to divide the potential into
    # coulomb_rng: Range of the coulomb potential (fm) (set to zero beyond)
    def __init__(self, Z, A, coulomb_rng, discr_steps=100):
        # TODO dictionary with atomic masses for mother and daughter nuclei
        self.mass_dict = Dict.empty(key_type=mass_dict_key_type, value_type=float64) # key: (Z, A)
        self.mass_dict[(92, 238)] = 221742.9 # U 238
        self.mass_dict[(90, 234)] = 218010.23 # Th 234 # TODO add other elements

        self.Z = Z
        self.A = A
        self.R = 1.3 * A ** (1/3) # TODO vary this?
        self.V0 = - 134 # MeV
        self.E_kin = self.E_kin_alpha()
        self.discr_steps = discr_steps
        self.coulomb_rng = coulomb_rng

        
def coulomb_minus_E(r, E, Z1, Z2):
        return fs * Z1 * Z2 * hc / r - E

def E_kin_alpha(Z, A):
        mass_dict = dict() # key: (Z, A)
        mass_dict[(92, 238)] = 221742.9 # U 238
        mass_dict[(90, 234)] = 218010.23 # Th 234 # TODO also add other elements here
        M_1 = mass_dict[(Z, A)]
        M_2 = mass_dict[(Z-2, A-4)]
        E_defect = M_1 - M_2 - M_alpha

        v_rel_squared = 2 * E_defect * (M_2 + M_alpha) / (M_2 * M_alpha) # (c)
        E_alpha = 0.5 * v_rel_squared * M_alpha 

        return E_alpha

def get_coulomb_range(Z, A):
    E = E_kin_alpha(Z, A)
    x = scp.optimize.fsolve(coulomb_minus_E, x0=1, args=(E, Z-2, 2))
    return x # TODO ??
