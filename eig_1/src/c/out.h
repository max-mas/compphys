//
// Created by max on 4/5/24.
//

#ifndef EIG1_OUT_H
#define EIG1_OUT_H

#include <Eigen/Dense>
#include <string>

const void evals_to_file(const Eigen::VectorXd & evals, const std::string & path);

const void evecs_to_file(const Eigen::MatrixXd & evecs, const std::string & path);

#endif //EIG1_OUT_H
