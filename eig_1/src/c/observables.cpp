//
// Created by max on 4/10/24.
//

#include "observables.h"
#include <complex>
#include <cmath>

Eigen::MatrixXd derivative_op(int num_bins, double x_max) {
    double h = 2 * x_max / double(num_bins - 1); // to ensure the last x point is x_max

    Eigen::MatrixXd ddz = Eigen::MatrixXd::Zero(num_bins, num_bins);

    for (int i = 0; i < num_bins; i++) {
        if (i > 1) { ddz(i, i-2) = 1 / (12 * h); }
        if (i > 0) { ddz(i, i-1) = -2  / (3  * h); }
        if (i < num_bins - 1) { ddz(i, i+1) =  2 / (3  * h); }
        if (i < num_bins - 2) { ddz(i, i+2) = -1 / (12 * h); }
    }

    return ddz;
}

Eigen::MatrixXcd momentum_op(int num_bins, double x_max) {
    Eigen::MatrixXcd p = derivative_op(num_bins, x_max);
    return - 1j * p; // units of sqrt(hbar m omega)
}

Eigen::MatrixXd position_op(int num_bins, double x_max) {
    Eigen::VectorXd x_diag = Eigen::VectorXd::Zero(num_bins);
    double h = 2 * x_max / double(num_bins - 1);
    for (int i = 0; i < num_bins; i++) {
        x_diag(i) = -x_max + i * h;
    }
    return x_diag.asDiagonal();
}

Eigen::MatrixXd a_dag_op(int num_bins, double x_max) {
    Eigen::MatrixXd ddz = derivative_op(num_bins, x_max);
    Eigen::MatrixXd z = position_op(num_bins, x_max);

    return 1/sqrt(2) * (z - ddz);
}

Eigen::MatrixXd a_op(int num_bins, double x_max) {
    Eigen::MatrixXd ddz = derivative_op(num_bins, x_max);
    Eigen::MatrixXd z = position_op(num_bins, x_max);

    return 1/sqrt(2) * (z + ddz);
}

Eigen::MatrixXd number_op(int num_bins, double x_max) {
    Eigen::MatrixXd a_dag = a_dag_op(num_bins, x_max);
    Eigen::MatrixXd a = a_op(num_bins, x_max);
    return a_dag * a;
}