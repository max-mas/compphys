//
// Created by max on 4/4/24.
//

#include "discrete_hamiltonian.h"
#include <cmath>

Eigen::MatrixXd discrete_hamiltonian(const double x_max,
                                           const long num_bins,
                                           const std::function<double (double)> & potential) {

    double h = 2 * x_max / double(num_bins - 1); // to ensure the last x point is x_max
    double h2 = pow(h, 2);
    Eigen::VectorXd discretized_potential(num_bins);
    for (int i = 0; i < num_bins; i++) {
        discretized_potential(i) = potential(-x_max + i * h); //evaluate V at x_i
    }

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_bins, num_bins);

    H(0, 0) =  1.0 / h2 + discretized_potential(0);
    H(0, 1) = -0.5 / h2;
    for (int i = 1; i < num_bins - 1; i++) {
        H(i, i-1) = -0.5 / h2;
        H(i, i)   =  1.0 / h2 + discretized_potential(i);
        H(i, i+1) = -0.5 / h2;

    }
    H(num_bins - 1, num_bins - 2) = -0.5 / h2;
    H(num_bins - 1, num_bins - 1) =  1.0 / h2 + discretized_potential(num_bins - 1);

    return H;
}

const std::vector<Eigen::VectorXd> discrete_hamiltonian_tridiagonals(const double x_max,
                                                               const long num_bins,
                                                               const std::function<double (double)>& potential) {
    double h = 2 * x_max / double(num_bins - 1); // to ensure the last x point is x_max
    double h2 = pow(h, 2);
    Eigen::VectorXd H_diag(num_bins), H_subdiag(num_bins - 1);
    for (int i = 0; i < num_bins; i++) {
        H_diag(i) = 1.0 / h2 + potential(-x_max + i * h); //evaluate V at x_i
        if (i < num_bins - 1) {
            H_subdiag(i) = -0.5 / h2;
        }
    }

    return std::vector<Eigen::VectorXd> {H_diag, H_subdiag};
}
