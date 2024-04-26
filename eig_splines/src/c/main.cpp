//
// Created by max on 4/24/24.
//

#include "spherical_seq.h"
#include "potentials.h"

#include <Eigen/Dense>

#include <iostream>
#include <string>

int main() {
    int N = 1200;
    Eigen::VectorXd points = Eigen::VectorXd::Zero(N);
    double delta = 1e-5;
    double exponent = 0;
    double end_exponent = 5;
    double exponent_step = (end_exponent - exponent) / (N - 1);
    for (int i = 1; i < points.size(); i++) {
        points(i) = points(i-1) + delta * pow(10, exponent);
        exponent += exponent_step;
    }
    std::cout << "Last point = " << points(Eigen::indexing::last) << std::endl;
    int l_max = 10;
    for (int l = 0; l <= l_max; l++) {
        spherical_seq<double> s(points, coulomb<double>, l);
        s.solve();
        std::cout << "l = " << l << " energies:" << std::endl;
        std::cout << s.energies << std::endl;
        int n_max = 0;
        for (double e: s.energies) {
            if (e < 0) {n_max++;} else break;
        }
        for (int n = 0; n <= n_max; n++) {
            std::string path_state = "../results/states/rad_wf_l" + std::to_string(l) + "_n"
                    + std::to_string(n+1+l) + "_m0.txt";
            s.save_solution_n(n, 1000, 1e-4, points(Eigen::indexing::last), path_state);
        }
        std::string path_energy = "../results/energies/energies_l" + std::to_string(l) + ".txt";
        s.save_energies(path_energy);
    }


    return 0;
}