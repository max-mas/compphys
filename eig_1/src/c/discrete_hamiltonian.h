//
// Created by max on 4/4/24.
//

#ifndef EIG1_DISCRETE_HAMILTONIAN_H
#define EIG1_DISCRETE_HAMILTONIAN_H

#include <Eigen/Dense>
#include <functional>

const Eigen::MatrixXd discrete_hamiltonian(const double x_max,
                                           const long num_bins,
                                           std::function<double (double)> potential);


#endif //EIG1_DISCRETE_HAMILTONIAN_H
