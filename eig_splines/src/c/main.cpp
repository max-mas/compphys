//
// Created by max on 4/24/24.
//

#include "spherical_seq.h"
#include "potentials.h"

#include <Eigen/Dense>

#include <iostream>

int main() {
    Eigen::VectorXd points = Eigen::VectorXd::Zero(500);
    for (int i = 1; i < points.size(); i++) {
        points(i) = points(i-1) + 0.0001 * exp(i/50);
    }
    std::cout << "Last point = " << points(Eigen::indexing::last) << std::endl;
    spherical_seq<double> s(points, coulomb<double>, 0);
    s.solve();
    std::cout << s.energies << std::endl;
    s.save_solution_n(0, 1000, points(Eigen::indexing::last), "../results/states/rad_wf_n1_l0_m0.txt");
    s.save_solution_n(1, 1000, points(Eigen::indexing::last), "../results/states/rad_wf_n2_l0_m0.txt");
    s.save_solution_n(2, 1000, points(Eigen::indexing::last), "../results/states/rad_wf_n3_l0_m0.txt");

    return 0;
}