//
// Created by max on 4/5/24.
//

#ifndef EIG1_OUT_H
#define EIG1_OUT_H

#include <Eigen/Dense>
#include <string>

/**
 * Saves eigenvalues to file.
 * @param evals Vector with eigenvalues.
 * @param path Path to write file to (including filename).
 */
const void evals_to_file(const Eigen::VectorXd & evals, const std::string & path);

/**
 * Saves eigenvectors to file, each row corresponds to the i-th eigenvector.
 * @param evals Matrix with eigenvectors in columns.
 * @param path Path to write file to (including filename).
 */
const void evecs_to_file(const Eigen::MatrixXd & evecs, const std::string & path);

#endif //EIG1_OUT_H
