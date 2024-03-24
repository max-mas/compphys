import numpy as np
import scipy as scp
from numba import int64, float64, complex64
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
    def E_kin_alpha(self) -> float:
        M_1 = self.mass_dict[(self.Z, self.A)]
        M_2 = self.mass_dict[(self.Z-2, self.A-4)]
        E_defect = M_1 - M_2 - M_alpha

        return E_defect


    # function that generates the discretised potential
    def piecewise_constant_potential(self) -> float64[:]: # type: ignore
        V = np.empty(self.discr_steps + 2, dtype=np.double)
        dx = (self.coulomb_rng - self.R) / self.discr_steps

        #first value of V is V0 (the well potential)
        V[0] = self.V0
        for i in range(1, self.discr_steps + 2):
            x = self.R + (i-1) * dx
            if np.abs(x - self.coulomb_rng) < 1e-10: # TODO is this valid? prevents that k outside is 0
                V[i] = self.coulomb(x + 1e-13) # offset
                continue
            V[i] = self.coulomb(x) # calculate coulomb potential at the connection points
        # last value of V is (almost) exactly E_kin if coulomb_rng was calculated correctly!

        return V
    
    # k / kappa in units of 1/fm
    # we use abs() to ensure the result is not complex in cases where E - V is almost 0
    def k(self, V) -> float:
        return np.sqrt(2 * M_alpha * np.abs(self.E_kin - V)) / hc
    
    def kappa(self, V) -> float:
        return np.sqrt(2 * M_alpha * np.abs(V - self.E_kin)) / hc


    # function that generates the matrix for the LSE
    def coeff_mat(self) -> complex64[:, :]: # type: ignore
        # connection points of the piecewise potential
        xs = np.ones(self.discr_steps + 1, dtype=np.double) * self.R  
        # potential bin separation
        dx = (self.coulomb_rng - self.R) / self.discr_steps
        for i in range(self.discr_steps + 1):
            xs[i] += i * dx

        V = self.piecewise_constant_potential() # potential
        
        # A is a complex square matrix
        A = np.zeros((2 * self.discr_steps + 3, 2 * self.discr_steps + 3), dtype=np.complex64)
        A[0, 0] = 1 # to fix the first constant to 1
        x = xs[0]
        V1 = V[0]
        V2 = V[1]
        # first block: oscillating and exponential solution meet
        # continuitiy of psi
        A[1, 0] =  np.exp( 1j * self.k(V1) * x)
        A[1, 1] =  np.exp(-1j * self.k(V1) * x)
        A[1, 2] = -np.exp( self.kappa(V2) * x)
        A[1, 3] = -np.exp(-self.kappa(V2) * x)
        # continuitiy of d_x psi
        A[2, 0] =  1j * self.k(V1) * np.exp( 1j * self.k(V1) * x)
        A[2, 1] = -1j * self.k(V1) * np.exp(-1j * self.k(V1) * x) 
        A[2, 2] = -     self.kappa(V2) * np.exp( self.kappa(V2) * x) 
        A[2, 3] =       self.kappa(V2) * np.exp(-self.kappa(V2) * x) 

        for i in range(3, 2*self.discr_steps + 1):
            j = (i-1) // 2  # j-th connection point
            V1 = V[j]
            V2 = V[j + 1]
            x = xs[j]
            
            # two exponential solutions meet
            if i % 2 == 1:
                # continuitiy of psi
                A[i, 2*j    ] =  np.exp( self.kappa(V1) * x)
                A[i, 2*j + 1] =  np.exp(-self.kappa(V1) * x)
                A[i, 2*j + 2] = -np.exp( self.kappa(V2) * x)
                A[i, 2*j + 3] = -np.exp(-self.kappa(V2) * x)
            else:
                # continuitiy of d_x psi
                A[i, 2*j]       =  self.kappa(V1) * np.exp( self.kappa(V1) * x)
                A[i, 2*j + 1]   = -self.kappa(V1) * np.exp(-self.kappa(V1) * x) 
                A[i, 2*j + 2]   = -self.kappa(V2) * np.exp( self.kappa(V2) * x) 
                A[i, 2*j + 3]   =  self.kappa(V2) * np.exp(-self.kappa(V2) * x) 
            
        x = xs[-1]
        V1 = V[-2]
        V2 = V[-1]
        # last connection point: exponential and oscillating solution meet
        # we fix the incoming plane wave to 0
        # continuitiy of psi
        A[-2, -3] =  np.exp( self.kappa(V1) * x)
        A[-2, -2] =  np.exp(-self.kappa(V1) * x)
        A[-2, -1] = -np.exp( 1j * self.k(V2) * x)
        # continuitiy of d_x psi
        A[-1, -3] =   self.kappa(V1) * np.exp( self.kappa(V1) * x) 
        A[-1, -2] = - self.kappa(V1) * np.exp(-self.kappa(V1) * x)
        A[-1, -1] = -1j * self.k(V2) * np.exp( 1j * self.k(V2) * x)

        return A

    # function that solves the LSE and returns the coefficients
    def solve(self) -> complex64[:]: # type: ignore
        if not self.solved:
            A = self.coeff_mat()
            b = np.zeros(2 * self.discr_steps + 3, dtype=np.complex64)
            b[0] = 1 # we fix A = 1 as suggested in class
            self.x = np.linalg.solve(A, b) # store solution coefficients for later use

            # check solution:
            err = A @ self.x - b
            zero = np.linalg.norm(err)
            if zero < 1e-5:
                self.solved = True # no need to do this multiple times

                return self.x 
            else: # throw error if solution is invalid
                print("||Ax - b|| =", zero)
                raise Exception(f"Numerical solution does not satisfy Ax - b = 0")

    # function that returns the transmission coefficient
    def get_transm_coeff(self) -> float:
        if not self.solved: # solve if not solved
            self.solve()

        V = self.piecewise_constant_potential() # potential
        k1 = self.k(V[0])
        k2 = self.k(V[-1])

        T = np.abs(self.x[-1]) ** 2 * k2 / k1
        return T
    
    # function that gets the half life from T
    def get_half_life(self) -> float: # seconds
        v = np.sqrt(2 * self.E_kin / M_alpha) * 2.998e23  # velocity (fm/s)
        tau = 2 * self.R / (v * self.get_transm_coeff()) # decay constant
        t_12 = tau * np.log(2) # half life
        
        return t_12 #/ (60 * 24 * 365) #years

    # function that builds the wave function from the coefficients
    # x range: 0 to coulomb range + 10 fm
    # n: number of evaluation points
    def calculate_psi(self, n=1000) -> complex64[:]: # type: ignore
        if not self.solved: # solve if not solved
            self.solve()
        
        xs = np.linspace(0.0, self.coulomb_rng + 10, n) # positions
        dx = (self.coulomb_rng - self.R) / self.discr_steps # potential bin separation
        V = self.piecewise_constant_potential() # potential
        psi = np.zeros(n, dtype=np.complex64) 

        for i, x in enumerate(xs):
            if x <= self.R: # oscillating solution at x < R
                psi[i] = self.x[0] * np.exp( 1j * self.k(V[0]) * x) + self.x[1] * np.exp( -1j * self.k(V[0]) * x)
            elif x >= self.coulomb_rng: # osillating solution at x > coulomb range
                psi[i] = self.x[-1] * np.exp( 1j * self.k(V[-1]) * x)
            else: # decaying solution in between
                j = int((x - self.R) / dx) + 1 # j-th potential bin    
                psi[i] = self.x[2*j] * np.exp(self.kappa(V[j]) * x) \
                    + self.x[2*j + 1] * np.exp(-self.kappa(V[j]) * x)
        
        self.psi = psi
        return psi
            
    
    # Get probability density (x)
    # n: Number of discrete positions
    def calculate_density(self, n) -> float64[:]: # type: ignore
        if not self.solved: # solve for coefficients if not done before
            self.solve()

        self.calculate_psi(n) # calculate psi at appropriate number of positions
        
        self.density = (self.psi.real ** 2 + self.psi.imag ** 2).astype(np.float64) # calculate |psi|^2
        
        return self.density

         

    # A: Atomic mass number
    # Z: Charge number
    # R: Core radius (fm)
    # V0: Depth of core well wrt. alpha particle energy (MeV)
    # discr_steps: Number of discrete steps to divide the potential into
    # coulomb_rng: Range of the coulomb potential (fm) (set to zero beyond)
    def __init__(self, Z, A, coulomb_rng, discr_steps=100, R_factor=1.35, V0 = -134):
        # TODO dictionary with atomic masses for mother and daughter nuclei
        self.mass_dict = Dict.empty(key_type=mass_dict_key_type, value_type=float64) # key: (Z, A)
        self.mass_dict[(92, 238)] = 221742.9  # U  238
        self.mass_dict[(90, 234)] = 218010.23 # Th 234
        self.mass_dict[(92, 235)] = 218942.03 # U  235
        self.mass_dict[(90, 231)] = 215208.95 # Th 231
        self.mass_dict[(90, 232)] = 216142.08 # Th 232
        self.mass_dict[(88, 228)] = 212409.6  # Ra 228
        self.mass_dict[(86, 222)] = 206808.06 # Rn 222
        self.mass_dict[(84, 218)] = 203074.07 # Po 218
        self.mass_dict[(84, 212)] = 197466.38 # Po 212
        self.mass_dict[(82, 208)] = 193729.02 # Pb 208


        self.Z = Z
        self.A = A
        self.R = R_factor * A ** (1/3)
        self.V0 = V0 # MeV
        self.E_kin = self.E_kin_alpha()
        self.discr_steps = discr_steps
        self.coulomb_rng = coulomb_rng # this must be given such that V(coulomb_rng) = E_kin !!
        if not np.abs(self.coulomb(coulomb_rng) - self.E_kin) < 1e-8:
            raise ValueError("Invalid coulomb range!") # raise error if range invalid

        
# To solve Coulomb - E_kin = 0
def coulomb_minus_E(r, E, Z1, Z2):
        return fs * Z1 * Z2 * hc / r - E

# Copy of above function accessible without creating an object
def E_kin_alpha(Z, A):
        mass_dict = dict() # key: (Z, A)
        mass_dict[(92, 238)] = 221742.9  # U 238
        mass_dict[(90, 234)] = 218010.23 # Th 234
        mass_dict[(92, 235)] = 218942.03 # U  235
        mass_dict[(90, 231)] = 215208.95 # Th 231
        mass_dict[(90, 232)] = 216142.08 # Th 232
        mass_dict[(88, 228)] = 212409.6  # Ra 228
        mass_dict[(86, 222)] = 206808.06 # Rn 222
        mass_dict[(84, 218)] = 203074.07 # Po 218
        mass_dict[(84, 212)] = 197466.38 # Po 212
        mass_dict[(82, 208)] = 193729.02 # Pb 208
        M_1 = mass_dict[(Z, A)]
        M_2 = mass_dict[(Z-2, A-4)]
        E_defect = M_1 - M_2 - M_alpha

        return E_defect

# Get range until which Coulomb > E_kin for given element
def get_coulomb_range(Z, A):
    E = E_kin_alpha(Z, A)
    x = scp.optimize.fsolve(coulomb_minus_E, x0=1, args=(E, Z-2, 2))
    return x
