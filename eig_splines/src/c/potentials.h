//
// Created by max on 4/24/24.
//

#ifndef EIG_SPLINES_POTENTIALS_H
#define EIG_SPLINES_POTENTIALS_H

#include <limits>

template <typename numeric_type>
numeric_type coulomb(numeric_type r) {
    // TODO UNITS!
    // prevent div by 0
    return (r != 0) ? -1/r : -1/std::numeric_limits<numeric_type>::min();
}

template <typename numeric_type>
numeric_type zero_fn(numeric_type r) {
    return 0.0;
}

#endif //EIG_SPLINES_POTENTIALS_H
