//
// Created by max on 4/4/24.
//

#ifndef EIG1_DISCRETE_HAMILTONIAN_H
#define EIG1_DISCRETE_HAMILTONIAN_H

#include <Eigen/Dense>
#include <functional>
#include <vector>

/**
 * Returns the discretized real-space representation of the hamiltonian for a 1d potential,
 * centered around the origin. Uses the three-point formula for d^2/dx^2.
 * @param x_max Maximum distance away from the origin.
 * @param num_bins Number of discrete positions.
 * @param potential Function that returns the potential. Must have correct signature.
 * @return Matrix representation of H.
 */
Eigen::MatrixXd discrete_hamiltonian(double x_max, long num_bins,
                                     const std::function<double (double)>& potential);

/**
 * Returns the discretized real-space representation of the hamiltonian for a 1d symmetric potential V(-x) = V(x),
 * for which parity is conserved so H can be seperated into two blocks. Uses the three-point formula for d^2/dx^2.
 * @param x_max Maximum distance away from the origin.
 * @param num_bins Number of discrete positions.
 * @param potential Function that returns the potential. Must have correct signature.
 * @return Vector with parity blocks of H (even first, odd second).
 */
std::vector<Eigen::MatrixXd> discrete_hamiltonian_parity(double x_max, long num_bins,
                                     const std::function<double (double)>& potential);

/**
 * Returns the discretized real-space representation of the hamiltonian for a 1d potential, centered around the origin.
 * Uses the five-point formula for d^2/dx^2.
 * @param x_max Maximum distance away from the origin.
 * @param num_bins Number of discrete positions.
 * @param potential Function that returns the potential. Must have correct signature.
 * @return Matrix representation of H.
 */
Eigen::MatrixXd discrete_hamiltonian_five_point(double x_max, long num_bins,
                                                const std::function<double (double)>& potential);

/**
 * Returns the discretized real-space representation of the hamiltonian for a 1d symmetric potential V(-x) = V(x),
 * for which parity is conserved so H can be seperated into two blocks. Uses the five-point formula for d^2/dx^2.
 * @param x_max Maximum distance away from the origin.
 * @param num_bins Number of discrete positions.
 * @param potential Function that returns the potential. Must have correct signature.
 * @return Vector with parity blocks of H (even first, odd second).
 */
std::vector<Eigen::MatrixXd> discrete_hamiltonian_five_point_parity(double x_max, long num_bins,
                                                const std::function<double (double)>& potential);

/**
 * Returns the diagonal and subdiagonal of the discretized real-space representation of the hamiltonian
 * for a 1d potential, centered around the origin. Must use the three-point formula for d^2/dx^2
 * to ensure the tridiagonal banded structure of H.
 * @param x_max Maximum distance away from the origin.
 * @param num_bins Number of discrete positions.
 * @param potential Function that returns the potential. Must have correct signature.
 * @return Vector with diagonal and subdiagonal of H.
 */
std::vector<Eigen::VectorXd> discrete_hamiltonian_tridiagonals(double x_max, long num_bins,
                                                               const std::function<double (double)>& potential);

/**
 * Returns the diagonal and subdiagonal of the discretized real-space representation of the hamiltonian
 * for a 1d symmetric potential V(-x) = V(x), for which parity is conserved so H can be seperated into two blocks.
 * Must use the three-point formula for d^2/dx^2 to ensure the tridiagonal banded structure of H.
 * @param x_max Maximum distance away from the origin.
 * @param num_bins Number of discrete positions.
 * @param potential Function that returns the potential. Must have correct signature.
 * @return Vector containing H_diag even, H_subdiag even, H_diag odd, H_subdiag odd in that order.
 */
std::vector<Eigen::VectorXd> discrete_hamiltonian_tridiagonals_parity(double x_max, long num_bins,
                                                                     const std::function<double (double)>& potential);

/**
 * Constructs the transformation matrix from position basis to a basis that encodes parity.
 * @param num_bins Number of discrete positions.
 * @return Transformation matrix
 */
Eigen::MatrixXd parity_transform(long num_bins);

#endif //EIG1_DISCRETE_HAMILTONIAN_H
