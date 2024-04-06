//
// Created by max on 4/6/24.
//

#ifndef EIG1_BENCHMARK_H
#define EIG1_BENCHMARK_H

#include <vector>
#include <string>
#include <functional>

#include <Eigen/Dense>

void benchmark_full_ed(const std::vector<int> & bin_numbers,
                       const std::vector<std::function<std::vector<Eigen::MatrixXd>
                               (double, long, const std::function<double (double)>&) >> & funcs,
                       const std::vector<std::string> & func_names,
                       const std::string & path);

#endif //EIG1_BENCHMARK_H
