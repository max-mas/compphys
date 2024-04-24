//
// Created by max on 4/24/24.
//

#ifndef EIG_SPLINES_POTENTIALS_H
#define EIG_SPLINES_POTENTIALS_H

template <typename numeric_type>
numeric_type coulomb(numeric_type r) {
    // TODO UNITS!
    // prevent div by 0
    return (r > 1e-15) ? -1/r : -1/(1e-15);
}

template <typename numeric_type>
numeric_type zero_fn(numeric_type r) {
    // TODO UNITS!
    // prevent div by 0
    return 0.0;
}

#endif //EIG_SPLINES_POTENTIALS_H
