//
// Created by max on 4/4/24.
//

#include "discrete_hamiltonian.h"

const Eigen::MatrixXd discrete_hamiltonian(const double x_max,
                                           const long num_bins,
                                           std::function<double (double)> potential) {

    double h = 2 * x_max / num_bins;
    Eigen::VectorXd discretized_positions(num_bins), discretized_potential(num_bins);
    for (int i = 0; i < num_bins; i++) {
        discretized_positions(i) = -x_max + i * h; // discretize x
        discretized_potential(i) = potential(discretized_positions(i)); //evaluate V at x_i
    }
    

}
