//
// Created by max on 4/5/24.
//

#include <Eigen/Eigenvalues>

//#include <openmp.h>

#include "solve.h"
#include "potentials.h"


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

const std::vector<Eigen::MatrixXd> solve_five_point_full(double x_max, long num_bins,
                                                    const std::function<double (double)>& potential) {
    Eigen::MatrixXd H = discrete_hamiltonian_five_point(x_max, num_bins, potential);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(H);
    return std::vector<Eigen::MatrixXd>({es.eigenvalues(), es.eigenvectors()});
}

Eigen::VectorXd solve_five_point_parity(double x_max, long num_bins,
                                        const std::function<double (double)>& potential) {
    std::vector<Eigen::MatrixXd> even_odd_H = discrete_hamiltonian_five_point_parity(x_max, num_bins, potential);
    Eigen::VectorXd evs = Eigen::VectorXd::Zero(num_bins);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_even;
    es_even.compute(even_odd_H[0]);
    evs(Eigen::seq(0, num_bins/2 - 1)) = es_even.eigenvalues();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_odd;
    es_odd.compute(even_odd_H[1]);
    evs(Eigen::seq(num_bins / 2, num_bins - 1)) = es_odd.eigenvalues();

    return evs;
}

const std::vector<Eigen::MatrixXd> solve_five_point_parity_full(double x_max, long num_bins,
                                                                const std::function<double (double)>& potential) {
    std::vector<Eigen::MatrixXd> even_odd_H = discrete_hamiltonian_five_point_parity(x_max, num_bins, potential);
    Eigen::VectorXd evs = Eigen::VectorXd::Zero(num_bins);
    Eigen::MatrixXd evecs = Eigen::MatrixXd::Zero(num_bins, num_bins);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < 2; i++) {
        if (i == 0) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_even;
            es_even.compute(even_odd_H[0]);
            evs(Eigen::seq(0, num_bins / 2 - 1)) = es_even.eigenvalues();
            evecs(Eigen::seq(0, num_bins / 2 - 1), Eigen::seq(0, num_bins / 2 - 1)) = es_even.eigenvectors();
        }
        else {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_odd;
            es_odd.compute(even_odd_H[1]);
            evs(Eigen::seq(num_bins / 2, num_bins - 1)) = es_odd.eigenvalues();
            evecs(Eigen::seq(num_bins / 2, num_bins - 1), Eigen::seq(num_bins / 2, num_bins - 1)) = es_odd.eigenvectors();
        }
    }

    { //transform evecs back, scoped to save memory
        Eigen::MatrixXd T = parity_transform(num_bins);
        evecs = T * evecs; // vector transform, not matrix transform!
    }
    //sort evecs back
    Eigen::MatrixXd evecs_sorted = Eigen::MatrixXd::Zero(num_bins, num_bins); // n/2 -> 1, n/2 + 1 -> 3, n/2 + 2 -> 5 ;; n - 1 -> n - 1
    // n-1 - n/2 + 1 =
    for (int i = 0; i < num_bins/2; i++) { evecs_sorted.col(2*i) = evecs.col(i); }
    for (int i = 0; i < num_bins/2; i++) { evecs_sorted.col(2*i + 1) = evecs.col(i + num_bins/2); }

    //sort evs back
    Eigen::VectorXd evs_sorted = Eigen::VectorXd::Zero(num_bins);
    for (int i = 0; i < num_bins/2; i++) {
        evs_sorted(2*i) = evs(i);
        evs_sorted(2*i + 1) = evs(num_bins/2 + i);
    }

    return std::vector<Eigen::MatrixXd>({evs_sorted, evecs_sorted});
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

const std::vector<Eigen::MatrixXd> solve_from_tridiag_parity_full(double x_max, long num_bins,
                                                                  const std::function<double (double)>& potential) {
    std::vector<Eigen::VectorXd> H = discrete_hamiltonian_tridiagonals_parity(x_max, num_bins, potential);
    Eigen::VectorXd & H_diag_even = H[0];
    Eigen::VectorXd & H_subdiag_even = H[1];
    Eigen::VectorXd & H_diag_odd = H[2];
    Eigen::VectorXd & H_subdiag_odd = H[3];

    Eigen::VectorXd evs = Eigen::VectorXd::Zero(num_bins);
    Eigen::MatrixXd evecs = Eigen::MatrixXd::Zero(num_bins, num_bins);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < 2; i++) {
        if (i == 0) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_even;
            es_even.computeFromTridiagonal(H_diag_even, H_subdiag_even);
            evs(Eigen::seq(0, num_bins/2 - 1)) = es_even.eigenvalues();
            evecs(Eigen::seq(0, num_bins/2 - 1), Eigen::seq(0, num_bins/2 - 1)) = es_even.eigenvectors();
        }
        else {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_odd;
            es_odd.computeFromTridiagonal(H_diag_odd, H_subdiag_odd);
            evs(Eigen::seq(num_bins / 2, num_bins - 1)) = es_odd.eigenvalues();
            evecs(Eigen::seq(num_bins / 2, num_bins - 1), Eigen::seq(num_bins / 2, num_bins - 1)) = es_odd.eigenvectors();
        }
    }

    { //transform evecs back, scoped to save memory
        Eigen::MatrixXd T = parity_transform(num_bins);
        evecs = T * evecs; // vector transform, not matrix transform!
    }
    //sort evecs back
    Eigen::MatrixXd evecs_sorted = Eigen::MatrixXd::Zero(num_bins, num_bins); // n/2 -> 1, n/2 + 1 -> 3, n/2 + 2 -> 5 ;; n - 1 -> n - 1
    // n-1 - n/2 + 1 =
    for (int i = 0; i < num_bins/2; i++) { evecs_sorted.col(2*i) = evecs.col(i); }
    for (int i = 0; i < num_bins/2; i++) { evecs_sorted.col(2*i + 1) = evecs.col(i + num_bins/2); }

    //sort evs back
    Eigen::VectorXd evs_sorted = Eigen::VectorXd::Zero(num_bins);
    for (int i = 0; i < num_bins/2; i++) {
        evs_sorted(2*i) = evs(i);
        evs_sorted(2*i + 1) = evs(num_bins/2 + i);
    }

    return std::vector<Eigen::MatrixXd>({evs_sorted, evecs_sorted});
}
