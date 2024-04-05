//
// Created by max on 4/5/24.
//

#ifndef EIG1_SOLVE_H
#define EIG1_SOLVE_H

#include "discrete_hamiltonian.h"
#include <vector>

Eigen::VectorXd solve_basic(double x_max, long num_bins,
                            const std::function<double (double)>& potential);

const std::vector<Eigen::MatrixXd> solve_basic_full(double x_max, long num_bins,
                                              const std::function<double (double)>& potential);

Eigen::VectorXd solve_from_tridiag(double x_max, long num_bins,
                                   const std::function<double (double)>& potential);

const std::vector<Eigen::MatrixXd> solve_from_tridiag_full(double x_max, long num_bins,
                                                     const std::function<double (double)>& potential);

#endif //EIG1_SOLVE_H
