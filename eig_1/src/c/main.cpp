//
// Created by max on 4/4/24.
//

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>

#include <Eigen/Dense>

#include "potentials.h"
#include "discrete_hamiltonian.h"
#include "solve.h"
#include "out.h"
#include "benchmark.h"
#include "inverse_iteration.h"
#include "observables.h"

int main() {
    bool get_best_full_diag = false;
    bool get_best_inv_it = false;
    bool bench_time = false;
    bool bench_xmax = false;
    bool playground = true;
    bool lennard_jones_test = false;

    if (get_best_full_diag) {
        std::vector<Eigen::MatrixXd> eval_evec = solve_five_point_parity_full(20, 4000, harmonic_bump_2);
        evals_to_file(eval_evec[0], "/home/max/code/compphys/eig_1/results/evs/evs_large_bump.txt");
        evecs_to_file(eval_evec[1], "/home/max/code/compphys/eig_1/results/evecs/evecs_large_bump.txt");
    }

    if (get_best_inv_it) {
        Eigen::MatrixXd H = discrete_hamiltonian_five_point(20, 4000, harmonic);
        std::vector<double> shifts({0, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3});
        std::vector<Eigen::MatrixXd> eval_evec = inverse_iteration(H, shifts);
        evals_to_file(eval_evec[0], "/home/max/code/compphys/eig_1/results/evs/evs_invit.txt");
        evecs_to_file(eval_evec[1], "/home/max/code/compphys/eig_1/results/evecs/evecs_invit.txt");
    }

    if (bench_time) {
        int avgs = 5;
        // Benchmark time
        std::vector<int> bin_nums({10, 50, 100, 200, 300, 500, 1000, 2000});
        std::vector<std::function<std::vector<Eigen::MatrixXd> (double, long, const std::function<double (double)>&) >>
                funcs({solve_basic_full, solve_from_tridiag_full, solve_from_tridiag_parity_full,
                       solve_five_point_full, solve_five_point_parity_full, get_gs_inverse_it});
        std::vector<std::string> func_names({"3 point", "3 point tridiag", "3 point tridiag parity",
                                             "5 point", "5 point parity", "Inv. it. (GS only)"});
        benchmark_full_ed(bin_nums, avgs, funcs, func_names, "/home/max/code/compphys/eig_1/results/bench/");
    }

    if (bench_xmax) {
        // Benchmark x_max
        int state = 50;

        int n_trials = 10; //next 10 here and n = 2500
        double max_x_max = 50;
        double x_max_curr = 1;
        double dx = (max_x_max - x_max_curr) / n_trials;
        int bins = 2500;
        std::vector<double> x_maxs(n_trials);

        std::generate(x_maxs.begin(), x_maxs.end(),
                      [&](){double x = x_max_curr; x_max_curr += dx; return x;});
        std::vector<std::function<std::vector<Eigen::MatrixXd> (double, long, const std::function<double (double)>&) >> funcs;
        std::vector<std::string> func_names;
        if (state == 0) {
            funcs = std::vector<std::function<std::vector<Eigen::MatrixXd>(double, long, const std::function<double (double)>&) >>
                    ({solve_from_tridiag_parity_full, solve_five_point_parity_full, get_gs_inverse_it});
            func_names = std::vector<std::string>({"3 point tridiag parity", "5 point parity", "Inv. it. (GS only)"});
        } else {
            funcs = std::vector<std::function<std::vector<Eigen::MatrixXd>(double, long, const std::function<double (double)>&) >>
                    ({solve_from_tridiag_parity_full, solve_five_point_parity_full});
            func_names = std::vector<std::string>({"3 point tridiag parity", "5 point parity"});
        }

        benchmark_excited_state_accuracy_x_max(x_maxs, bins, state, funcs, func_names, "/home/max/code/compphys/eig_1/results/bench/");
    }

    if (playground) {
        int num_bins = 4000;
        double x_max = 20;
        std::vector<Eigen::MatrixXd> eval_evec = solve_five_point_parity_full(x_max, num_bins, harmonic);
        Eigen::MatrixXd evec = eval_evec[1];
        //Eigen::MatrixXcd p = momentum_op(num_bins, x_max);
        Eigen::MatrixXd x = position_op(num_bins, x_max);
        Eigen::MatrixXd number = number_op(num_bins, x_max);
        //Eigen::MatrixXd a_dag = a_dag_op(num_bins, x_max);

        //Eigen::VectorXd gs = evec.col(0);
        //Eigen::VectorXd state_1 = (a_dag * gs).normalized();
        //double test = (evec.col(1) - state_1).norm();
        //std::cout << test << std::endl;


        //Eigen::VectorXcd momenta = Eigen::VectorXcd::Zero(num_bins);
        //Eigen::VectorXcd squared_momenta = Eigen::VectorXcd::Zero(num_bins);
        //Eigen::VectorXcd positions = Eigen::VectorXcd::Zero(num_bins);
        Eigen::VectorXd squared_positions = Eigen::VectorXd::Zero(num_bins);
        Eigen::VectorXd numbers = Eigen::VectorXd::Zero(num_bins);
        for (int i = 0; i < num_bins; i++) {
            //momenta(i) = evec.col(i).transpose() * p * evec.col(i);
            //positions(i) = evec.col(i).transpose() * x * evec.col(i);
            squared_positions(i) = evec.col(i).transpose() * x*x * evec.col(i);
            //squared_momenta(i) = evec.col(i).transpose() * p*p * evec.col(i);
            numbers(i) = evec.col(i).transpose() * number * evec.col(i);
        }
        //std::cout << momenta << std::endl;
        //std::cout << positions << std::endl;
        //std::cout << squared_positions << std::endl;
        //std::cout << squared_momenta << std::endl;
        //std::cout << numbers << std::endl;
        evals_to_file(squared_positions,
                      "/home/max/code/compphys/eig_1/results/squared_positions.txt");
        evals_to_file(numbers,
                      "/home/max/code/compphys/eig_1/results/occupation.txt");


    }

    if (lennard_jones_test) {
        std::vector<Eigen::MatrixXd> eval_evec = solve_five_point_full(20, 1000, morse_potential);
        evals_to_file(eval_evec[0], "/home/max/code/compphys/eig_1/results/evs/evs_morse_1000.txt");
        evecs_to_file(eval_evec[1], "/home/max/code/compphys/eig_1/results/evecs/evecs_morse_1000.txt");
    }

    return 0;
}