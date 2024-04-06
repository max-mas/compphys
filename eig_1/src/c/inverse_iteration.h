//
// Created by max on 4/6/24.
//

#ifndef EIG1_INVERSE_ITERATION_H
#define EIG1_INVERSE_ITERATION_H

#include <vector>
#include <tuple>
#include <Eigen/Dense>

/**
 * Performs inverse iteration for a given matrix and a given list of shifts.
 * @param A Square matrix to find eigenpairs for.
 * @param shifts Vector of shifts to apply, this amounts to guesses of eigenvalues.
 * @param maxit Maximum number of iterations per shift.
 * @param epsilon Iteration stops if |(A - shift * I) x_i - x_i| < epsilon.
 * @return Vector containing an Eigen::VectorXd with the eigenvalues and an Eigen::MatrixXd with the eigenvectors
 * in the columns.
 */
std::vector<Eigen::MatrixXd> inverse_iteration(const Eigen::MatrixXd & A, const std::vector<double> & shifts,
                                               int maxit = 50, double epsilon = 1e-8);

/**
 * Performs inverse iteration for a Hamiltonian that is divided into two parity sectors (convenience function).
 * @param H_even_odd Vector of parity sectors.
 * @param shifts Vector of shifts to apply, this amounts to guesses of eigenvalues.
 * @param maxit Maximum number of iterations per shift.
 * @param epsilon Iteration stops if |(A - shift * I) x_i - x_i| < epsilon.
 * @return Vector containing an Eigen::VectorXd with the eigenvalues and an Eigen::MatrixXd with the eigenvectors
 * in the columns.
 */
std::vector<Eigen::MatrixXd> inverse_iteration_parity(const std::vector<Eigen::MatrixXd> & H_even_odd,
                                                      const std::vector<double> & shifts,
                                                      int maxit = 50, double epsilon = 1e-8);

#endif //EIG1_INVERSE_ITERATION_H
