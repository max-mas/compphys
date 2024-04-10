//
// Created by max on 4/10/24.
//

#ifndef EIG1_OBSERVABLES_H
#define EIG1_OBSERVABLES_H

#include <Eigen/Dense>

Eigen::MatrixXd derivative_op(int num_bins, double x_max);

Eigen::MatrixXcd momentum_op(int num_bins, double x_max);

Eigen::MatrixXd position_op(int num_bins, double x_max);

Eigen::MatrixXd a_dag_op(int num_bins, double x_max);

Eigen::MatrixXd a_op(int num_bins, double x_max);

Eigen::MatrixXd number_op(int num_bins, double x_max);

#endif //EIG1_OBSERVABLES_H
