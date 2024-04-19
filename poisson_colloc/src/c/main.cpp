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
    for (int i = 0; i < b.knot_points.size() - 4; i++) { //
        b.save_B_i(i, 1000, -1, 11, "../results/B_i_k/");
        b.save_B_i(i, 1000, -1, 11, "../results/B_i_k_x/", 1);
        b.save_B_i(i, 1000, -1, 11, "../results/B_i_k_xx/", 2);
    }
    return 0;
}
