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
                       int avgs,
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
            for (int avg = 0; avg < avgs; avg ++) {
                // get current time
                const auto start = std::chrono::steady_clock::now();
                // do work
                std::vector<Eigen::MatrixXd> eval_evec = funcs[method_index](x_max, bin_numbers[size_index], harmonic);
                // get end time
                const auto end = std::chrono::steady_clock::now();
                // duration
                const std::chrono::duration<double, std::nano> duration_ns = end - start;
                // duration as float in seconds
                double duration_s_double = duration_ns.count() / 1e9;
                times[method_index][size_index] += duration_s_double / avgs;
                // exact GS erg is 0.5
                gs_erg_acc[method_index][size_index] = std::abs(0.5 - eval_evec[0](0));

                std::cout   << "N = " << bin_numbers[size_index] << ", T = " << duration_s_double
                            << "s, GS erg error: " << gs_erg_acc[method_index][size_index] << std::endl;
            }
        }
    }

    // save times
    std::ofstream file;
    file.open(path + "bench_time_avgs_" + std::to_string(avgs) + ".txt");
    for (int n : bin_numbers) {file << n << ",";}
    file << std::endl;
    for (const std::vector<double>& time_vec: times) {
        for (double time : time_vec) {
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) //precision!
                 << std::scientific << time << ",";
        }
        file << std::endl;
    }
    file.close();

    // save GS erg errors
    std::ofstream file2;
    file2.open(path + "bench_gs_erg_err.txt");
    for (int n : bin_numbers) {file2 << n << ",";}
    file2 << std::endl;
    for (const std::vector<double>& gs_errs: gs_erg_acc) {
        for (double err : gs_errs) {
            file2 << std::setprecision(std::numeric_limits<long double>::digits10 + 1) //precision!
                 << std::scientific << err << ",";
        }
        file2 << std::endl;
    }
    file2.close();
}

void benchmark_excited_state_accuracy_x_max(const std::vector<double> & x_maxs, int bin_number, int state,
                                            const std::vector<std::function<std::vector<Eigen::MatrixXd>
                                                    (double, long, const std::function<double (double)>&) >> & funcs,
                                            const std::vector<std::string> & func_names,
                                            const std::string & path) {
    int n_trials = x_maxs.size();
    std::vector<std::vector<double>> excited_erg_acc(funcs.size(), std::vector<double>(n_trials, 0.0));

    std::cout << "Benchmarking." << std::endl;
    for (int method_index = 0; method_index < funcs.size(); method_index++) {
        std::cout << "Method: " << func_names[method_index] << std::endl;
        for (int range_index = 0; range_index < n_trials; range_index++) {
            double x_max = x_maxs[range_index];
            // get current time
            const auto start = std::chrono::steady_clock::now();
            // do work
            std::vector<Eigen::MatrixXd> eval_evec = funcs[method_index](x_max, bin_number, harmonic);
            // get end time
            const auto end = std::chrono::steady_clock::now();
            // duration
            const std::chrono::duration<double, std::nano> duration_ns = end - start;
            // duration as float in seconds
            double duration_s_double = duration_ns.count() / 1e9;
            // exact GS erg is 0.5
            excited_erg_acc[method_index][range_index] = std::abs(0.5 + state - eval_evec[0](state));

            std::cout   << "x_max = " << x_max << ", T = " << duration_s_double
                        << "s, state " << state << " error: " << excited_erg_acc[method_index][range_index] << std::endl;
        }
    }

    // save excited state erg errors
    std::ofstream file;
    file.open(path + "bench_xmax_err_state" + std::to_string(state) + "_n_" + std::to_string(bin_number) + ".txt");
    for (double x_max : x_maxs) {file << x_max << ",";}
    file << std::endl;
    for (const std::vector<double>& gs_errs: excited_erg_acc) {
        for (double err : gs_errs) {
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) //precision!
                 << std::scientific << err << ",";
        }
        file << std::endl;
    }
    file.close();
}
