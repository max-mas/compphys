//
// Created by max on 4/6/24.
//

#include "inverse_iteration.h"

#include <iostream>

#include "solve.h"

std::vector<Eigen::MatrixXd>
inverse_iteration(const Eigen::MatrixXd &A, const std::vector<double> &shifts, int maxit, double epsilon) {
    int n = A.cols();
    int n_shifts = shifts.size();

    Eigen::VectorXd evals(n_shifts);
    Eigen::MatrixXd evecs(n, n_shifts);

    for (int i_shift = 0; i_shift < n_shifts; i_shift ++) {
        double lambda = shifts[i_shift];
        Eigen::MatrixXd A_shifted = A - lambda * Eigen::MatrixXd::Identity(n, n); // shift matrix
        Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXd>> LU(A_shifted); // this operates in place, rms coefficients

        Eigen::VectorXd x = Eigen::VectorXd::Random(n); // good initial guess?
        x.normalize();
        for (int j = 0; j < maxit; j++) {
            Eigen::VectorXd x_new = LU.solve(x);
            x_new.normalize(); // normalise in every step to prevent blow up

            double change = (x - x_new).norm(); // if this settles to zero, convergence is reached

            x = x_new;

            if (change < epsilon) {
                break;
            } else if (j == maxit - 1) { // don't want to make this an exception b/c I don't want to handle them
                std::cout   << "Warning: Inverse iteration did not converge for shift " << lambda << " and "
                            << maxit << " iterations. Last change in norm: " << change << std::endl;
            }
        }
        evals(i_shift) = x.transpose() * A * x;
        evecs.col(i_shift) = x;
    }

    return std::vector<Eigen::MatrixXd>({evals, evecs});
}

std::vector<Eigen::MatrixXd> inverse_iteration_parity(const std::vector<Eigen::MatrixXd> & H_even_odd,
                                                      const std::vector<double> & shifts,
                                                      int maxit, double epsilon) {
    int n_dim = H_even_odd[0].cols() + H_even_odd[1].cols();
    int n_shifts = shifts.size();
    int n_eigenpairs = 2 * n_shifts;
    // note: in this way, we get 2 eigenpairs for every shift, one from every sector
    Eigen::VectorXd evals(2 * n_shifts);
    Eigen::MatrixXd evecs = Eigen::MatrixXd::Zero(n_dim, 2*n_shifts);

#pragma omp parallel for num_threads(2) // brrr
    for (int i = 0; i < 2; i++) {
        if (i == 0) {
            std::vector<Eigen::MatrixXd> eval_evec_even = inverse_iteration(H_even_odd[0], shifts, maxit, epsilon);
            evals(Eigen::seq(0, n_eigenpairs/2 - 1)) = eval_evec_even[0];
            evecs(Eigen::seq(0, n_dim/2 - 1), Eigen::seq(0, n_eigenpairs/2 - 1)) = eval_evec_even[1];
        }
        else {
            std::vector<Eigen::MatrixXd> eval_evec_odd  = inverse_iteration(H_even_odd[1], shifts, maxit, epsilon);
            evals(Eigen::seq(n_eigenpairs/2, n_eigenpairs - 1)) = eval_evec_odd[0];
            evecs(Eigen::seq(n_dim/2, n_dim - 1), Eigen::seq(n_eigenpairs/2, n_eigenpairs - 1)) = eval_evec_odd[1];
        }
    }

    { //transform evecs back, scoped to save memory
        Eigen::MatrixXd T = parity_transform(n_dim);
        evecs = T * evecs; // vector transform, not matrix transform!
    }

    return std::vector<Eigen::MatrixXd>({evals, evecs}); // unsorted!
}