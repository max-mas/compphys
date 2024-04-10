//
// Created by max on 4/4/24.
//

#ifndef EIG1_POTENTIALS_H
#define EIG1_POTENTIALS_H

#include <cmath>

/**
 * Harmonic potential.
 * @param z Dimensionless distance from the origin.
 * @return V in units of hbar omega.
 */
double harmonic(double z);

/**
 * Modified harmonic potential.
 * @param z Dimensionless distance from the origin.
 * @return V in units of hbar omega.
 */
double harmonic_bump(double z);

double harmonic_bump_2(double z);

double lennard_jones(double z);

double morse_potential(double z);

#endif //EIG1_POTENTIALS_H
