//
// Created by max on 5/7/24.
//
#include <iostream>
#include <filesystem>
#include <string>

#include "atom.h"

int main() {
    int maxit = 80;
    int n_seq = 150;
    int n_pot = 300;
    double rmax = 40; //TODO might be too small for bigger atoms

    std::ofstream file;
    file.open("../results/ionergs3.txt");

    //std::vector<unsigned int> Zs({2, 10});
    auto Zs = Eigen::VectorXi::LinSpaced(89, 20, 108);
    std::cout << Zs << std::endl;
    auto ionisation_ergs = std::vector<double>();
    for (auto Z: Zs) {
        auto Es = std::vector<double>();
        for (unsigned int N_e = Z; N_e >= Z-1; N_e -= 1) {
            atom<double> A(Z, N_e, maxit, rmax, n_seq, n_pot);
            A.run();
            std::filesystem::create_directory("../results/atoms3/" + std::to_string(Z) + "/") ;
            A.save_rho("../results/atoms3/" + std::to_string(Z) + "/rho_ne_" + std::to_string(N_e) + ".txt");
            A.save_summary("../results/atoms3/" + std::to_string(Z) + "/summary_ne_" + std::to_string(N_e) + ".txt");
            A.save_potentials("../results/atoms3/" + std::to_string(Z) + "/");
            Es.emplace_back(A.total_energy());
            std::cout << "------------------------------------------------------------------------------------" << std::endl;
        }
        std::cout << "Ionization energy for Z = " << Z << ": " << abs(Es[1] - Es[0]) << std::endl;
        ionisation_ergs.emplace_back(abs(Es[1] - Es[0]));
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
             << Z << "," << std::scientific << abs(Es[1] - Es[0]) << std::endl;
        std::cout << "------------------------------------------------------------------------------------" << std::endl;
    }

    file.close();

    return 0;
}