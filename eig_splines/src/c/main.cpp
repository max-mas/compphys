/**
 * @file main.cpp
 * @author Max Maschke (m.maschke@tu-bs.de)
 * @brief Main
 * @version 0.1
 * @date 2024-04-28
 * 
 * @copyright Copyright (c) 2024 Max Maschke
 * 
 */

#include "spherical_seq.h"
#include "potentials.h"

#include <Eigen/Dense>

#include <iostream>
#include <string>
#include <tuple>
#include <algorithm>

int main() {
    if (false) {
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
    }
    if (true) {
        int nn = 20;
        Eigen::VectorXi ns = Eigen::VectorXi::LinSpaced(nn, 5, 100);
        std::vector<std::pair<int, int>> num_bound_states;
        std::vector<std::pair<double, double>> gs_err;
        for (int n : ns) {
            Eigen::VectorXd points = Eigen::VectorXd::LinSpaced(n, 0, 100);
            std::cout << "n = " << n << ", testing num. of bound states in l = 0 sector." << std::endl
                      << "Linspaced points [0, " << points(Eigen::indexing::last) << "]: " << std::flush;
            spherical_seq<double> s(points, coulomb<double>, 0);
            s.solve();
            int num_linspaced = std::count_if(s.energies.begin(), s.energies.end(),
                                           [](double e){return e < 0;});
            double gs_err_linspaced = abs(-0.5 - s.energies(0));
            std::cout << num_linspaced << " bound states. GS erg err: " << gs_err_linspaced << std::endl;
            points = Eigen::VectorXd::Zero(n);
            double exponent = -6;
            double end_exponent = 2;
            double exponent_step = (end_exponent - exponent) / (n - 2);
            for (int i = 1; i < points.size(); i++) {
                points(i) = pow(10, exponent);
                exponent += exponent_step;
            }
            std::cout << "Exp spaced points [0, " << points(Eigen::indexing::last) << "]: " << std::flush;
            s = spherical_seq<double>(points, coulomb<double>, 0);
            s.solve();
            int num_exp = std::count_if(s.energies.begin(), s.energies.end(),
                                              [](double e){return e < 0;});
            double gs_err_exp = abs(-0.5 - s.energies(0));
            std::cout << num_exp << " bound states. GS erg err: " << gs_err_exp << std::endl;
            num_bound_states.emplace_back(std::pair<int, int>{num_linspaced, num_exp});
            gs_err.emplace_back(std::pair<double, double>{gs_err_linspaced, gs_err_exp});
        }
        std::ofstream file;
        file.open("../results/ns2.txt");
        // write
        for (int j = 0; j < nn; j++) {
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
                 << std::scientific << ns[j] << ","
                 << std::scientific << num_bound_states[j].first << ","
                 << std::scientific << num_bound_states[j].second << std::endl;
        }
        // don't forget to clean up :)
        file.close();
        file.open("../results/gs_err2.txt");
        // write
        for (int j = 0; j < nn; j++) {
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
                 << std::scientific << ns[j] << ","
                 << std::scientific << gs_err[j].first << ","
                 << std::scientific << gs_err[j].second << std::endl;
        }
        // don't forget to clean up :)
        file.close();
    }


    return 0;
}