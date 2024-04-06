//
// Created by max on 4/4/24.
//

#include "discrete_hamiltonian.h"
#include <cmath>
#include <stdexcept>

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

    for (int i = 0; i < num_bins; i++) {
        if (i > 0) { H(i, i-1) = -0.5 / h2; }
        H(i, i)   =  1.0 / h2 + discretized_potential(i);
        if (i < num_bins - 1) { H(i, i+1) = -0.5 / h2; };

    }

    return H;
}

std::vector<Eigen::MatrixXd> discrete_hamiltonian_parity(double x_max, long num_bins,
                                                         const std::function<double (double)>& potential) {
    if (num_bins % 2 != 0) {
        throw std::runtime_error("For parity basis to be unique, num_bins must be even.");
    }

    Eigen::MatrixXd H_blockdiag = Eigen::MatrixXd::Zero(num_bins, num_bins);

    { //local scope to free memory of H and T ASAP
        Eigen::MatrixXd T = parity_transform(num_bins);
        Eigen::MatrixXd H = discrete_hamiltonian(x_max, num_bins, potential);
        H_blockdiag = T.transpose() * H * T;
    }
    std::vector<Eigen::MatrixXd> even_sec_odd_sec({Eigen::MatrixXd::Zero(num_bins/2, num_bins/2),
                                                   Eigen::MatrixXd::Zero(num_bins/2, num_bins/2)});

    even_sec_odd_sec[0] = H_blockdiag(Eigen::seq(0, num_bins/2 -1), Eigen::seq(0, num_bins/2 -1));
    even_sec_odd_sec[1] = H_blockdiag(Eigen::seq(num_bins/2, num_bins-1), Eigen::seq(num_bins/2, num_bins-1));

    return even_sec_odd_sec;
}

Eigen::MatrixXd discrete_hamiltonian_five_point(double x_max, long num_bins,
                                                const std::function<double (double)>& potential) {
    double h = 2 * x_max / double(num_bins - 1); // to ensure the last x point is x_max
    double h2 = pow(h, 2);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_bins, num_bins);

    for (int i = 0; i < num_bins; i++) {
        if (i > 1) { H(i, i-2) = 0.5 / (12 * h2); }
        if (i > 0) { H(i, i-1) = -2  / (3  * h2); }

        H(i, i) = 2.5 / (2 * h2) + potential(-x_max + i * h);

        if (i < num_bins - 1) { H(i, i+1) =  -2 / (3  * h2); }
        if (i < num_bins - 2) { H(i, i+2) = 0.5 / (12 * h2); }
    }

    return H;
}

std::vector<Eigen::MatrixXd> discrete_hamiltonian_five_point_parity(double x_max, long num_bins,
                                                                          const std::function<double (double)>& potential) {
    if (num_bins % 2 != 0) {
        throw std::runtime_error("For parity basis to be unique, num_bins must be even.");
    }

    Eigen::MatrixXd H_blockdiag = Eigen::MatrixXd::Zero(num_bins, num_bins);

    { //local scope to free memory of H and T ASAP
        Eigen::MatrixXd T = parity_transform(num_bins);
        Eigen::MatrixXd H = discrete_hamiltonian_five_point(x_max, num_bins, potential);
        H_blockdiag = T.transpose() * H * T;
    }
    std::vector<Eigen::MatrixXd> even_sec_odd_sec({Eigen::MatrixXd::Zero(num_bins/2, num_bins/2),
                                                      Eigen::MatrixXd::Zero(num_bins/2, num_bins/2)});

    even_sec_odd_sec[0] = H_blockdiag(Eigen::seq(0, num_bins/2 -1), Eigen::seq(0, num_bins/2 -1));
    even_sec_odd_sec[1] = H_blockdiag(Eigen::seq(num_bins/2, num_bins-1), Eigen::seq(num_bins/2, num_bins-1));

    return even_sec_odd_sec;
}

std::vector<Eigen::VectorXd> discrete_hamiltonian_tridiagonals(const double x_max,
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

std::vector<Eigen::VectorXd> discrete_hamiltonian_tridiagonals_parity
    (double x_max, long num_bins, const std::function<double (double)>& potential) {

    std::vector<Eigen::MatrixXd> even_sec_odd_sec = discrete_hamiltonian_parity(x_max, num_bins, potential);
    Eigen::VectorXd H_diag_even(num_bins/2), H_diag_odd(num_bins/2);
    Eigen::VectorXd H_subdiag_even(num_bins - 1), H_subdiag_odd(num_bins - 1);

    for (int i = 0; i < num_bins/2; i++) {
        H_diag_even(i) = even_sec_odd_sec[0](i, i);
        H_diag_odd(i) = even_sec_odd_sec[1](i, i);
        if (i < num_bins/2 - 1) {
            H_subdiag_even(i) = even_sec_odd_sec[0](i+1, i);
            H_subdiag_odd(i) = even_sec_odd_sec[1](i+1, i);
        }
    }

    return std::vector<Eigen::VectorXd> {H_diag_even, H_subdiag_even, H_diag_odd, H_subdiag_odd};
}

Eigen::MatrixXd parity_transform(long num_bins) {
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(num_bins, num_bins);
    for (int i = 0; i < num_bins / 2; i++) {
        T(i, i) = 1 / std::sqrt(2); //even sector
        T(num_bins - i - 1, i) = 1 / std::sqrt(2); // 1 / sqrt(2) normalises each column

        T(i, i + num_bins/2) = 1 / std::sqrt(2); //odd sector
        T(num_bins - i - 1, i + num_bins/2) = -1 / std::sqrt(2);
    }

    return T;
}
