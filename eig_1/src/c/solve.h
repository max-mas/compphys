//
// Created by max on 4/5/24.
//

#ifndef EIG1_SOLVE_H
#define EIG1_SOLVE_H

#include "discrete_hamiltonian.h"
#include <vector>

/**
 * Solves the discretised Schrödinger equation, provides energies and states.
 * Uses three point formula.
 * @param x_max Maximum distance from the origin.
 * @param num_bins Number of discretised points.
 * @param potential Potential function, must have correct signature.
 * @return Vector containing a vector of eigenvalues and the matrix with the eigenvectors.
 */
const std::vector<Eigen::MatrixXd> solve_basic_full(double x_max, long num_bins,
                                              const std::function<double (double)>& potential);

/**
 * Solves the discretised Schrödinger equation, provides energies and states.
 * Uses five point formula.
 * @param x_max Maximum distance from the origin.
 * @param num_bins Number of discretised points.
 * @param potential Potential function, must have correct signature.
 * @return Vector containing a vector of eigenvalues and the matrix with the eigenvectors.
 */
const std::vector<Eigen::MatrixXd> solve_five_point_full(double x_max, long num_bins,
                                                         const std::function<double (double)>& potential);

/**
 * Solves the discretised Schrödinger equation, provides energies and states.
 * Uses five point formula and parity sectors.
 * @param x_max Maximum distance from the origin.
 * @param num_bins Number of discretised points.
 * @param potential Potential function, must have correct signature.
 * @return Vector containing a vector of eigenvalues and the matrix with the eigenvectors.
 */
const std::vector<Eigen::MatrixXd> solve_five_point_parity_full(double x_max, long num_bins,
                                                           const std::function<double (double)>& potential);

/**
 * Solves the discretised Schrödinger equation, provides energies and states.
 * Uses three point formula, solves from banded structure of H.
 * @param x_max Maximum distance from the origin.
 * @param num_bins Number of discretised points.
 * @param potential Potential function, must have correct signature.
 * @return Vector containing a vector of eigenvalues and the matrix with the eigenvectors.
 */
const std::vector<Eigen::MatrixXd> solve_from_tridiag_full(double x_max, long num_bins,
                                                     const std::function<double (double)>& potential);

/**
 * Solves the discretised Schrödinger equation, provides energies and states.
 * Uses three point formula, solves from banded structure of H. Also uses parity sectors.
 * @param x_max Maximum distance from the origin.
 * @param num_bins Number of discretised points.
 * @param potential Potential function, must have correct signature.
 * @return Vector containing a vector of eigenvalues and the matrix with the eigenvectors.
 */
const std::vector<Eigen::MatrixXd> solve_from_tridiag_parity_full(double x_max, long num_bins,
                                                           const std::function<double (double)>& potential);

/**
 * Solves the discretised Schrödinger equation, provides only ground state and energy.
 * Uses five point formula and parity sectors. Uses inverse iteration.
 * @param x_max Maximum distance from the origin.
 * @param num_bins Number of discretised points.
 * @param potential Potential function, must have correct signature.
 * @return Vector containing a vector of eigenvalues and the matrix with the eigenvectors.
 */
const std::vector<Eigen::MatrixXd> get_gs_inverse_it(double x_max, long num_bins,
                                                     const std::function<double (double)>& potential);

#endif //EIG1_SOLVE_H
