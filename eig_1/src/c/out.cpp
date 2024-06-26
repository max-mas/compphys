//
// Created by max on 4/5/24.
//

#include "out.h"
#include <fstream>
#include <iostream>
#include <iomanip>

const void evals_to_file(const Eigen::VectorXd & evals, const std::string & path) {
    std::ofstream file;
    file.open(path);

     //get all digts
    for (double ev: evals) {
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
             << std::scientific << ev << std::endl;
    }

    file.close();
}

const void evecs_to_file(const Eigen::MatrixXd & evecs, const std::string & path) {
    std::ofstream file;
    file.open(path);

    int n_vecs = evecs.cols();
    int n_dim = evecs.rows();

    //get all digts
    for (int i = 0; i < n_vecs; i++) {
        for (int j = 0; j < n_dim; j++) {
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
                 << std::scientific << evecs(j, i) << ",";
        }
        file << std::endl;
    }

    file.close();
}