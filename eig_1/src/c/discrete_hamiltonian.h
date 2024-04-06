//
// Created by max on 4/4/24.
//

#ifndef EIG1_DISCRETE_HAMILTONIAN_H
#define EIG1_DISCRETE_HAMILTONIAN_H

#include <Eigen/Dense>
#include <functional>
#include <vector>

Eigen::MatrixXd discrete_hamiltonian(double x_max, long num_bins,
                                     const std::function<double (double)>& potential);

std::vector<Eigen::MatrixXd> discrete_hamiltonian_parity(double x_max, long num_bins,
                                     const std::function<double (double)>& potential);

Eigen::MatrixXd discrete_hamiltonian_five_point(double x_max, long num_bins,
                                                const std::function<double (double)>& potential);

std::vector<Eigen::MatrixXd> discrete_hamiltonian_five_point_parity(double x_max, long num_bins,
                                                const std::function<double (double)>& potential);

std::vector<Eigen::VectorXd> discrete_hamiltonian_tridiagonals(double x_max, long num_bins,
                                                               const std::function<double (double)>& potential);

std::vector<Eigen::VectorXd> discrete_hamiltonian_tridiagonals_parity(double x_max, long num_bins,
                                                                     const std::function<double (double)>& potential);

Eigen::MatrixXd parity_transform(long num_bins);

#endif //EIG1_DISCRETE_HAMILTONIAN_H
