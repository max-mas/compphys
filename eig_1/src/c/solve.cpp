//
// Created by max on 4/5/24.
//

#include "solve.h"
#include "potentials.h"
#include <Eigen/Eigenvalues>

Eigen::VectorXd solve_basic(double x_max, long num_bins,
                            const std::function<double (double)>& potential) {
    Eigen::MatrixXd H = discrete_hamiltonian(x_max, num_bins, potential);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(H, Eigen::EigenvaluesOnly);
    return es.eigenvalues();
}

const std::vector<Eigen::MatrixXd> solve_basic_full(double x_max, long num_bins,
                                                    const std::function<double (double)>& potential) {
    Eigen::MatrixXd H = discrete_hamiltonian(x_max, num_bins, potential);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(H);
    return std::vector<Eigen::MatrixXd>({es.eigenvalues(), es.eigenvectors()});
}

Eigen::VectorXd solve_from_tridiag(double x_max, long num_bins,
                                   const std::function<double (double)>& potential) {
    std::vector<Eigen::VectorXd> A = discrete_hamiltonian_tridiagonals(x_max, num_bins, potential);
    Eigen::VectorXd & H_diag = A[0];
    Eigen::VectorXd & H_subdiag = A[1];
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.computeFromTridiagonal(H_diag, H_subdiag, Eigen::EigenvaluesOnly);
    return es.eigenvalues();
}

const std::vector<Eigen::MatrixXd> solve_from_tridiag_full(double x_max, long num_bins,
                                                           const std::function<double (double)>& potential) {
    std::vector<Eigen::VectorXd> A = discrete_hamiltonian_tridiagonals(x_max, num_bins, potential);
    Eigen::VectorXd & H_diag = A[0];
    Eigen::VectorXd & H_subdiag = A[1];
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.computeFromTridiagonal(H_diag, H_subdiag);
    return std::vector<Eigen::MatrixXd>({es.eigenvalues(), es.eigenvectors()});
}
