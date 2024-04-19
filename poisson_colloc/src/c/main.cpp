//
// Created by max on 4/19/24.
//

#include "b_splines.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::VectorXd knots(11);
    knots << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

    b_splines<double> b(4, knots, true);
    std::cout << b.knot_points << std::endl;
    std::cout << b.B_i_k(3, 4, 0.5);
    return 0;
}
