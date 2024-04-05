//
// Created by max on 4/4/24.
//

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "potentials.h"
#include "discrete_hamiltonian.h"
#include "solve.h"
#include "out.h"

int main() {
    //Eigen::VectorXd ev1 = solve_basic(10, 500, harmonic);
    std::vector<Eigen::MatrixXd> eval_evec = solve_from_tridiag_full(10, 1000, harmonic);
    //std::cout << ev2 << std::endl;
    //std::cout << ev1 << std::endl;
    evals_to_file(eval_evec[0], "/home/max/code/compphys/eig_1/results/evs/evs_test.txt");
    evecs_to_file(eval_evec[1], "/home/max/code/compphys/eig_1/results/evecs/evecs_test.txt");
    return 0;
}