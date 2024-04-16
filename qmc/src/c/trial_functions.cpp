//
// Created by max on 4/11/24.
//

#include "trial_functions.h"
#include <stdexcept>

double harmonic_trial(std::vector<double> alpha_x) {
    double alpha = alpha_x[0];
    double x = alpha_x[1];
    return exp(-alpha * pow(x, 2));
}

double harmonic_local_erg(std::vector<double> alpha_x) {
    double alpha = alpha_x[0];
    double x = alpha_x[1];
    return alpha + pow(x, 2) * (0.5 - 2 * pow(alpha, 2));
}

double harmonic_trial_2d_2p(std::vector<double> alpha_x) {
    double lambda = 1.0;
    double alpha = alpha_x[0];
    double x1 = alpha_x[1];
    double y1 = alpha_x[2];
    double x2 = alpha_x[3];
    double y2 = alpha_x[4];

    double s = sqrt( pow(x1 - x2, 2) + pow(y1 - y2, 2) );

    return exp(-(pow(x1, 2) + pow(y1, 2) + pow(x2, 2) + pow(y2, 2)) / 2)
           * exp((lambda * s) / (1 + alpha * s));
}

double harmonic_local_erg_2d_2p(std::vector<double> alpha_x) { //TODO optimise function calls
    double lambda = 1.0;
    double alpha = alpha_x[0];
    double x1 = alpha_x[1];
    double y1 = alpha_x[2];
    double x2 = alpha_x[3];
    double y2 = alpha_x[4];

    double s = sqrt( pow(x1 - x2, 2) + pow(y1 - y2, 2) );
    double k1 = s * pow(1 + alpha * s, 2);
    double k2 = pow(s, 3) * pow(1 + alpha * s, 3);

    double electronic_term = lambda / s;
    double harmonic_term = 0.5 * (pow(x1, 2) + pow(y1, 2) + pow(x2, 2) + pow(y2, 2));

    double kinetic_term_x1 = -0.5 * (-1 + lambda / k1 - lambda * pow(x1 - x2, 2) / k2 * (1 + 3 * alpha * s)
              + pow(-x1 + lambda * (x1 - x2) / k1 , 2) );
    double kinetic_term_x2 = -0.5 * (-1 + lambda / k1 - lambda * pow(x2 - x1, 2) / k2 * (1 + 3 * alpha * s)
              + pow(-x2 + lambda * (x2 - x1) / k1 , 2) );
    double kinetic_term_y1 = -0.5 * (-1 + lambda / k1 - lambda * pow(y1 - y2, 2) / k2 * (1 + 3 * alpha * s)
              + pow(-y1 + lambda * (y1 - y2) / k1 , 2) );
    double kinetic_term_y2 = -0.5 * (-1 + lambda / k1 - lambda * pow(y2 - y1, 2) / k2 * (1 + 3 * alpha * s)
              + pow(-y2 + lambda * (y2 - y1) / k1 , 2) );

    return kinetic_term_x1 + kinetic_term_x2 + kinetic_term_y1 + kinetic_term_y2 + harmonic_term + electronic_term;
}