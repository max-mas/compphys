//
// Created by max on 4/11/24.
//
#include <iostream>
#include <tuple>

#include "monte_carlo_1d.h"
#include "trial_functions.h"


int main() {
    if (false) {
        std::pair<double, double> alpha_range = {0, 1};
        monte_carlo_1d testobject = monte_carlo_1d(1, 20, int(1e8),
                                                   harmonic_trial,harmonic_local_erg,
                                                   alpha_range, 0.01);
        testobject.run();
    }
    if (true) {
        std::pair<double, double> alpha_range = {0, 10};
        monte_carlo_1d testobject = monte_carlo_1d(1, 20, int(1e6),
                                                   harmonic_trial_2d_2p,harmonic_local_erg_2d_2p,
                                                   alpha_range, 0.01);
        testobject.run();
    }
}
