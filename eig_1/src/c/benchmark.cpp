//
// Created by max on 4/6/24.
//

#include "benchmark.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

#include "solve.h"
#include "potentials.h"

void benchmark_full_ed(const std::vector<int> & bin_numbers,
                       const std::vector<std::function<std::vector<Eigen::MatrixXd>
                               (double, long, const std::function<double (double)>&) >> & funcs,
                       const std::vector<std::string> & func_names,
                       const std::string & path) {
    double x_max = 10;
    int n_trials = bin_numbers.size();

    // store runtimes in each subvector
    std::vector<std::vector<double>> times(funcs.size(), std::vector<double>(n_trials, 0.0));
    std::vector<std::vector<double>> gs_erg_acc(funcs.size(), std::vector<double>(n_trials, 0.0));

    std::cout << "Benchmarking." << std::endl;
    for (int method_index = 0; method_index < funcs.size(); method_index++) {
        std::cout << "Method: " << func_names[method_index] << std::endl;
        for (int size_index = 0; size_index < n_trials; size_index++) {
            const auto start = std::chrono::steady_clock::now();
            std::vector<Eigen::MatrixXd> eval_evec = funcs[method_index](x_max, bin_numbers[size_index], harmonic);
            const auto end = std::chrono::steady_clock::now();
            const std::chrono::duration<double, std::nano> duration_ns = end - start;
            double duration_s_double = duration_ns.count() / 1e9;//seconds
            times[method_index][size_index] = duration_s_double;
            gs_erg_acc[method_index][size_index] = std::abs(0.5 - eval_evec[0](0));
            std::cout   << "N = " << bin_numbers[size_index] << ", T = " << duration_s_double
                        << "s, GS erg error: " << gs_erg_acc[method_index][size_index] << std::endl;
        }
    }

    std::ofstream file;
    file.open(path + "bench_time.txt");
    for (int n : bin_numbers) {file << n << ",";}
    file << std::endl;
    for (const std::vector<double>& time_vec: times) {
        for (double time : time_vec) {
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                 << std::scientific << time << ",";
        }
        file << std::endl;
    }
    file.close();

    std::ofstream file2;
    file.open(path + "bench_gs_erg_err.txt");
    for (int n : bin_numbers) {file << n << ",";}
    file2 << std::endl;
    for (const std::vector<double>& gs_errs: gs_erg_acc) {
        for (double err : gs_errs) {
            file2 << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                 << std::scientific << err << ",";
        }
        file2 << std::endl;
    }
    file2.close();
}
