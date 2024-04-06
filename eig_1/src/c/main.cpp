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

int main() {
    //std::vector<Eigen::MatrixXd> eval_evec = solve_from_tridiag_parity_full(10, 1000, harmonic);
    //evals_to_file(eval_evec[0], "/home/max/code/compphys/eig_1/results/evs/evs_test.txt");
    //evecs_to_file(eval_evec[1], "/home/max/code/compphys/eig_1/results/evecs/evecs_test.txt");


    std::vector<int> bin_nums({10, 50, 100, 200, 300, 500, 1000, 2000, 3000});
    std::vector<std::function<std::vector<Eigen::MatrixXd> (double, long, const std::function<double (double)>&) >>
            funcs({solve_basic_full, solve_from_tridiag_full, solve_from_tridiag_parity_full,
                   solve_five_point_full, solve_five_point_parity_full});
    std::vector<std::string> func_names({"3 point", "3 point tridiag", "3 point tridiag parity",
                                         "5 point", "5 point parity"});
    benchmark_full_ed(bin_nums, funcs, func_names, "/home/max/code/compphys/eig_1/results/bench/");

    return 0;
}