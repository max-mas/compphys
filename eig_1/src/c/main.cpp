//
// Created by max on 4/4/24.
//

#include <iostream>
#include <Eigen/Dense>
#include "potentials.h"

int main() {
    std::cout << "Hello world!" << std::endl;
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 0.0;
    std::cout << m(0, 0) << std::endl;
    std::cout << harmonic(2) << std::endl;
    std::cout << harmonic_bump(0) << std::endl;
    return 0;
}