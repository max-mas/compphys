//
// Created by max on 4/4/24.
//

#include "potentials.h"

const double harmonic(const double z) {
    return 0.5 * std::pow(z, 2);
}

const double harmonic_bump(const double z) {
    return 0.5 * std::pow(z, 2) + 2 * std::exp(-10.0 * std::pow(z, 2));
}
