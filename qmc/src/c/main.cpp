//
// Created by max on 4/11/24.
//
#include <iostream>
#include <tuple>


#include "monte_carlo.h"
#include "trial_functions.h"


int main() {
    if (false) {
        std::pair<double, double> alpha_range = {0, 1};
        monte_carlo mc = monte_carlo(1, 1, 30, int(1e7),
                                                   harmonic_trial,harmonic_local_erg,
                                                   alpha_range, 0.1, true,
                                                   "../results/1d");
        mc.run();
    }
    if (true) {
        std::pair<double, double> alpha_range = {0, 1};
        monte_carlo mc = monte_carlo(1, 1, 30, int(1e7),
                                     harmonic_trial,harmonic_local_erg,
                                     alpha_range, 0.05, true,
                                     "../results/1d_spaced", true);
        mc.run();
    }
    if (false) {
        std::pair<double, double> alpha_range = {0, 1};
        monte_carlo mc = monte_carlo(2, 2, 30, int(1e7),
                                                   harmonic_trial_2d_2p,harmonic_local_erg_2d_2p,
                                                   alpha_range, 0.1, true,
                                                   "../results/2d");
        mc.run();
    }
    if (false) {
        std::pair<double, double> alpha_range = {0, 1};
        monte_carlo mc = monte_carlo(2, 2, 30, int(1e7),
                                     harmonic_trial_2d_2p,harmonic_local_erg_2d_2p,
                                     alpha_range, 0.1, true,
                                     "../results/2d_spaced_larger_step", true);
        mc.run();
    }
}
