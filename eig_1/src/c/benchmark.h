//
// Created by max on 4/6/24.
//

#ifndef EIG1_BENCHMARK_H
#define EIG1_BENCHMARK_H

#include <vector>
#include <string>
#include <functional>

#include <Eigen/Dense>

/**
 * Benchmarks different eigensolvers and saves runtime and ground state error to file.
 * @param bin_numbers Number of discrete positions.
 * @param funcs Vector of functions that solve the eigenproblem (must have correct signature!)
 * @param func_names Names of given functions for console output.
 * @param path Directory to which to write results.
 */
void benchmark_full_ed(const std::vector<int> & bin_numbers,
                       int avgs,
                       const std::vector<std::function<std::vector<Eigen::MatrixXd>
                               (double, long, const std::function<double (double)>&) >> & funcs,
                       const std::vector<std::string> & func_names,
                       const std::string & path);

void benchmark_excited_state_accuracy_x_max(const std::vector<double> & x_maxs, int bin_number, int state,
                                            const std::vector<std::function<std::vector<Eigen::MatrixXd>
                                                    (double, long, const std::function<double (double)>&) >> & funcs,
                                            const std::vector<std::string> & func_names,
                                            const std::string & path);

#endif //EIG1_BENCHMARK_H
