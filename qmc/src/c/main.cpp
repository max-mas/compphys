//
// Created by max on 4/11/24.
//
#include <iostream>
#include <fstream>
#include <iomanip>
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
    if (false) {
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
                                                   "../results/2d_lambda2");
        mc.run();
    }
    if (false) {
        std::pair<double, double> alpha_range = {0, 1};
        monte_carlo mc = monte_carlo(2, 2, 30, int(1e7),
                                     harmonic_trial_2d_2p,harmonic_local_erg_2d_2p,
                                     alpha_range, 0.1, true,
                                     "../results/2d_spaced_lambda2", true);
        mc.run();
    }
    if (true) {
        std::vector<double> ls;
        std::vector<double> Es;
        std::vector<double> alphas;
        for (double l = 0; l <= 10; l = l+1) {
            GLOBAL_LAMBDA = l;
            std::cout << "Lambda = " << GLOBAL_LAMBDA << std::endl;
            std::pair<double, double> alpha_range = {0, 1};
            monte_carlo mc = monte_carlo(2, 2, 20, int(1e7),
                                         harmonic_trial_2d_2p,harmonic_local_erg_2d_2p,
                                         alpha_range, 0.1);
            mc.run();
            std::tuple<double, double, double> res = mc.get_best_trial();
            ls.push_back(GLOBAL_LAMBDA);
            Es.push_back(std::get<1>(res));
            alphas.push_back(std::get<0>(res));
        }
        std::ofstream file;
        file.open("../results/lambda_sweep.txt");
        for (int i = 0; i < ls.size(); i++) {
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
                 << std::scientific << ls[i] << ","
                 << std::scientific << alphas[i] << ","
                 << std::scientific << Es[i] << std::endl;
        }
        file.close();
    }
}
