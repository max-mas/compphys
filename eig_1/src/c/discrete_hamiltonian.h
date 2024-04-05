//
// Created by max on 4/4/24.
//

#ifndef EIG1_DISCRETE_HAMILTONIAN_H
#define EIG1_DISCRETE_HAMILTONIAN_H

#include <Eigen/Dense>
#include <functional>
#include <vector>

Eigen::MatrixXd discrete_hamiltonian(  double x_max,
                                       long num_bins,
                                       const std::function<double (double)>& potential);

const std::vector<Eigen::VectorXd> discrete_hamiltonian_tridiagonals(double x_max,
                                                               long num_bins,
                                                               const std::function<double (double)>& potential);

#endif //EIG1_DISCRETE_HAMILTONIAN_H
