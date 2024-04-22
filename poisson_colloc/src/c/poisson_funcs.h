//
// Created by max on 4/20/24.
//

#ifndef POISSON_POISSON_FUNCS_H
#define POISSON_POISSON_FUNCS_H

#include <cmath>

template <typename numeric_type>
numeric_type zero_function(numeric_type x) {
    return 0;
}

// domain: R+
template <typename numeric_type>
numeric_type phi_rhs_solid_sphere(numeric_type r) {
    numeric_type R = numeric_type(1.0);
    numeric_type rho = numeric_type(1.0);
    numeric_type pi = numeric_type(M_PI);
    if (r <= R) {
        return - 4 * pi * r * rho;
    } else {
        return numeric_type(0.0);
    }
}

// domain: R+
template <typename numeric_type>
numeric_type phi_rhs_shell(numeric_type r) {
    numeric_type R_outer = numeric_type(1.0);
    numeric_type R_inner = numeric_type(0.8);
    numeric_type rho = numeric_type(1.0);
    numeric_type pi = numeric_type(M_PI);
    if (r <= R_outer and r >= R_inner) {
        return - 4 * pi * r * rho;
    } else {
        return numeric_type(0.0);
    }
}

// domain: R+
template <typename numeric_type>
numeric_type phi_rhs_hydrogen(numeric_type r) {
    numeric_type a_0 = numeric_type(1.0);
    numeric_type e = numeric_type(1.0);
    numeric_type pi = numeric_type(M_PI);
    return - 4 * pi * r * e / (pi * pow(a_0, 3)) * exp(-2 * r / a_0);
}

// domain: R+
template <typename numeric_type>
numeric_type phi_rhs_hydrogen_2s(numeric_type r) {
    numeric_type a_0 = numeric_type(1.0);
    numeric_type e = numeric_type(1.0);
    numeric_type pi = numeric_type(M_PI);
    return - 4 * pi * r * e / (32.0 * pi * pow(a_0, 3)) * pow(2 - r/a_0, 2) * exp(-r / a_0);
}


#endif //POISSON_POISSON_FUNCS_H
