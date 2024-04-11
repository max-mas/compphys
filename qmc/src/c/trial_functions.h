//
// Created by max on 4/11/24.
//

#ifndef QMC_TRIAL_FUNCTIONS_H
#define QMC_TRIAL_FUNCTIONS_H

#include <cmath>
#include <vector>

double harmonic_trial(std::vector<double> alpha_x);

double harmonic_local_erg(std::vector<double> alpha_x);

double harmonic_trial_2d_2p(std::vector<double> alpha_x);

double harmonic_local_erg_2d_2p(std::vector<double> alpha_x);


#endif //QMC_TRIAL_FUNCTIONS_H
