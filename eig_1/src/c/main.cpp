//
// Created by max on 4/4/24.
//

#include <iostream>
#include <vector>
#include <string>
#include <functional>

#include <Eigen/Dense>

#include "potentials.h"
#include "discrete_hamiltonian.h"
#include "solve.h"
#include "out.h"
#include "benchmark.h"
#include "inverse_iteration.h"

int main() {
    std::vector<Eigen::MatrixXd> eval_evec = solve_five_point_parity_full(20, 4000, harmonic);
    evals_to_file(eval_evec[0], "/home/max/code/compphys/eig_1/results/evs/evs_test.txt");
    evecs_to_file(eval_evec[1], "/home/max/code/compphys/eig_1/results/evecs/evecs_test.txt");


    /*
    // Benchmark
    std::vector<int> bin_nums({10, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000});
    std::vector<std::function<std::vector<Eigen::MatrixXd> (double, long, const std::function<double (double)>&) >>
            funcs({solve_basic_full, solve_from_tridiag_full, solve_from_tridiag_parity_full,
                   solve_five_point_full, solve_five_point_parity_full, get_gs_inverse_it});
    std::vector<std::string> func_names({"3 point", "3 point tridiag", "3 point tridiag parity",
                                         "5 point", "5 point parity", "Inv. it. (GS only)"});
    benchmark_full_ed(bin_nums, funcs, func_names, "/home/max/code/compphys/eig_1/results/bench/");
     */

    /*
    std::vector<Eigen::MatrixXd> H_even_odd = discrete_hamiltonian_five_point_parity(10, 1000, harmonic);
    std::vector<double> shifts({0});
    std::vector<Eigen::MatrixXd> eval_evec = inverse_iteration_parity(H_even_odd, shifts);
    evals_to_file(eval_evec[0], "/home/max/code/compphys/eig_1/results/evs/evs_invit.txt");
    evecs_to_file(eval_evec[1], "/home/max/code/compphys/eig_1/results/evecs/evecs_invit.txt");
     */

    return 0;
}