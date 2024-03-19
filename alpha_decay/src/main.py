import numpy as np
import scipy as scp


def main():
    return


class Alphadecay:
    # TODO unit conversion constants as constant class variables

    # TODO dictionary with atomic masses for mother and daughter nuclei

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

        return


if __name__ == "main":
    main()