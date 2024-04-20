//
// Created by max on 4/19/24.
//

#include "b_splines.h"
#include "collocation.h"
#include "poisson_funcs.h"

#include <Eigen/Dense>
#include <iostream>

int main() {
    if (false) {
        Eigen::VectorXd knots(11);
        knots << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

        b_splines<double> b(4, knots);
        for (int i = 0; i < b.knot_points.size() - 4; i++) { //
            b.save_B_i(i, 1000, -1, 11, "../results/B_i_k/");
            b.save_B_i(i, 1000, -1, 11, "../results/B_i_k_x/", 1);
            b.save_B_i(i, 1000, -1, 11, "../results/B_i_k_xx/", 2);
        }
    }
    if (true) {
        double charge = 4.0/3.0 * M_PI * pow(1, 3) * 1.0;
        std::pair<double, double> interval({0, 5});
        std::pair<double, double> boundary({0, charge});
        int num_pts = 500;
        collocation<double> c(zero_function<double>,
                              zero_function<double>,
                              phi_rhs_solid_sphere<double>,
                              interval,
                              boundary,
                              num_pts);
        c.solve();
        c.save_solution(0, 5, 500, "../results/solution/solution_solidsphere.txt");
    }


    return 0;
}
