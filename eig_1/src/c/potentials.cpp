//
// Created by max on 4/4/24.
//

#include "potentials.h"

double harmonic(double z) {
    return 0.5 * std::pow(z, 2);
}

double harmonic_bump(double z) {
    return 0.5 * std::pow(z, 2) + 1 * std::exp(-10.0 * std::pow(z, 2));
}

double harmonic_bump_2(double z) {
    return 0.5 * std::pow(z, 2) + 5 * std::exp(-10.0 * std::pow(z, 2));
}

double lennard_jones(double z) {
    double z_max = 20; // ugh
    z = z + z_max + 1e-8; //prevent 1/0
    return 0.5 * std::pow(z, -12) - 0.5 * std::pow(z, +6);
}

double morse_potential(double z) {
    double z_max = 20; // ugh
    z = z + z_max + 1e-8; //prevent 1/0
    return pow(1 - exp((-z - 5)), 2);
}