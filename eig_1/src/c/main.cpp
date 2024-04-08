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

int main() {
    bool get_best_full_diag = true;
    bool get_best_inv_it = false;
    bool bench_time = false;
    bool bench_xmax = false;

    if (get_best_full_diag) {
        std::vector<Eigen::MatrixXd> eval_evec = solve_five_point_parity_full(20, 4000, harmonic);
        evals_to_file(eval_evec[0], "/home/max/code/compphys/eig_1/results/evs/evs_full_4000.txt");
        evecs_to_file(eval_evec[1], "/home/max/code/compphys/eig_1/results/evecs/evecs_full_4000.txt");
    }

    if (get_best_inv_it) {
        std::vector<Eigen::MatrixXd> H_even_odd = discrete_hamiltonian_five_point_parity(10, 1000, harmonic);
        std::vector<double> shifts({0});
        std::vector<Eigen::MatrixXd> eval_evec = inverse_iteration_parity(H_even_odd, shifts);
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


    return 0;
}