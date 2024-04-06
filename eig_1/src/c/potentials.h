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
const double harmonic(const double z);

/**
 * Modified harmonic potential.
 * @param z Dimensionless distance from the origin.
 * @return V in units of hbar omega.
 */
const double harmonic_bump(const double z);

#endif //EIG1_POTENTIALS_H
