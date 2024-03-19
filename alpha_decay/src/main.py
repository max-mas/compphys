import numpy as np
import scipy as scp
from numba import njit, int32, int64, float32, float64
from numba.types import UniTuple, DictType
mass_dict_key_type = UniTuple(int64, 2)
mass_dict_type = DictType(mass_dict_key_type, float64)
from numba.experimental import jitclass
from numba.typed import Dict

# TODO unit conversion constants as constant class variables:
hc = 197.3269631 # hbar * c (MeV * fm)
fs = 7.2973525693e-3 # fine structure constant (dimensionless)
M_alpha = 3727.379 #  alpha particle mass (MeV)



def main():
    a = Alphadecay(1, 1, 1, 1, 1, 1)
    print(a.test())
    return


@jitclass
class Alphadecay:    
    # typing of member variables
    A: int
    Z: int
    R: float
    V0: float
    discr_steps: int
    coulomb_rng: float    
    mass_dict: mass_dict_type

    # exact coulomb
    # Z1, Z2: charge numbers
    # r: distance (fm)
    # return: energy (MeV)
    def coulomb(self, Z1, Z2, r) -> float:
        return fs * Z1 * Z2 * hc / r
    
    # TODO rm
    def test(self) -> float:
        return self.mass_dict[(90, 234)]

    # TODO function that generates the discretised potential

    # TODO function that makes the matrix for the LSE

    # TODO function that solves the LSE and returns the coefficients

    # TODO function that builds the wave function from the coefficients

    # TODO plotting

    # TODO numerics testing

    # A: Atomic mass number
    # Z: Charge number
    # R: Core radius (fm)
    # V0: Depth of core well wrt. alpha particle energy TODO (unit)
    # discr_steps: Number of discrete steps to divide the potential into
    # coulomb_rng: Range of the coulomb potential (fm) (set to zero beyond)
    def __init__(self, A, Z, R, V0, discr_steps=100, coulomb_rng=10): #TODO change default range to sensible value
        self.A = A
        self.Z = Z
        self.R = R
        self.V0 = V0
        self.discr_steps = discr_steps
        self.coulomb_rng = coulomb_rng

        # TODO dictionary with atomic masses for mother and daughter nuclei
        self.mass_dict = Dict.empty(key_type=mass_dict_key_type, value_type=float64) # key: (Z, A)
        self.mass_dict[(92, 238)] = 221742.9 # U 238
        self.mass_dict[(90, 234)] = 218010.23 # Th 234


if __name__ == "__main__":
    main()