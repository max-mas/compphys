//
// Created by max on 4/11/24.
//

#include "monte_carlo.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

monte_carlo::monte_carlo(const unsigned int particleNum,
                               unsigned int spatialDim,
                               unsigned int trialNum,
                               unsigned int stepNum,
                               const std::function<double(std::vector<double>)> &trialWavefunction,
                               const std::function<double(std::vector<double>)> &localEnergy,
                               const std::pair<double, double> & alphaRange,
                               double step_stddev,
                               bool saveTrials,
                               std::string outPath,
                               bool spacedAlphas):
    trial_num(trialNum),
    spatial_dim(spatialDim),
    particle_num(particleNum),
    step_num(stepNum),
    trial_wavefunction(trialWavefunction),
    local_energy(localEnergy),
    alpha_range(alphaRange),
    save_trials(saveTrials),
    out_path(std::move(outPath)),
    spaced_alphas(spacedAlphas)
{
    // init rng
    normal_step = std::normal_distribution<double>(0.0, step_stddev);
    // randomly init positions
    positions = std::vector<double>(spatial_dim*particle_num, 0.0);
    std::generate(positions.begin(), positions.end(), [&](){return normal_step(gen);});
    // init energy(alpha) vector with zeros
    alpha_energy_std = std::vector<std::tuple<double, double, double>>(trial_num, {0, 0, 0});

    optimised_alpha_range = alpha_range;
}

double monte_carlo::select_next_alpha() {
    static double alpha_b = 0;
    static double alpha_x = 0;
    if (trial == 0) {
        return alpha_range.first;
    } else if (trial == 1) {
        return alpha_range.second;
    } else if (trial == 2) {
        alpha_b = golden_split(alpha_range.first, alpha_range.second);
        return alpha_b;
    } else if (trial == 3) {
        alpha_x = golden_split(alpha_range.first, std::get<0>(alpha_energy_std[2]));
        return alpha_x;
    }

    if (std::get<1>(alpha_energy_std[trial-1]) < std::get<1>(alpha_energy_std[trial-2])) {
        optimised_alpha_range.second = alpha_b;
        alpha_b = alpha_x;
        alpha_x = golden_split(optimised_alpha_range.first, alpha_b);
    } else {
        optimised_alpha_range.first = optimised_alpha_range.second;
        optimised_alpha_range.second = alpha_x;
        alpha_x = golden_split(optimised_alpha_range.first, alpha_b);
    }
    return alpha_x;
}

double monte_carlo::next_alpha_spaced() {
    double d_alpha = (alpha_range.second - alpha_range.first) / (trial_num-1);
    static double next_alpha = 0;
    static bool flag = false;
    if (!flag) {
        flag = true;
        return next_alpha;
    } else {
        next_alpha += d_alpha;
        return next_alpha;
    }
}

void monte_carlo::run() {
    if (has_run) {
        throw std::runtime_error("To run a new set of trials, create a new object!");
    }
    for (trial = 0; trial < trial_num; trial++) { //trial is a class member!
        if (!spaced_alphas) {
            alpha = select_next_alpha();
        } else {
            alpha = next_alpha_spaced();
        }

        std::cout << "Trial " << trial << ", doing " << step_num << " steps. alpha = " << alpha << std::flush;
        double energy = do_trial();
        std::cout << "; E = " << energy << ", std = " << stddev_energy << std::endl;
        alpha_energy_std[trial] = {alpha, energy, stddev_energy};
        if (save_trials) {save_trial();}
    }
    has_run = true;
    std::tuple<double, double, double> best_alpha_E = get_best_trial();
    std::cout << "Best trial: alpha = " << std::get<0>(best_alpha_E) << ", E = " << std::get<1>(best_alpha_E)
        << " std = " << std::get<2>(best_alpha_E) << std::endl;
    if (save_trials) {save_summary();}
    std::cout   << "Highest confidence alpha range: [" << optimised_alpha_range.first << ", "
                << optimised_alpha_range.second << "]" << std::endl;
}

double monte_carlo::do_trial() {
    reset_positions();
    current_energy = 0;
    stddev_energy = 0;
    std::uniform_real_distribution<double> zero_one_uniform(0, 1);
    unsigned int skip_initial = int(step_num) / 10; //TODO wasteful?
    unsigned int actual_steps = (step_num - skip_initial) * particle_num * spatial_dim; // important
    energies = std::vector<double>(actual_steps, 0);

    int energy_index = 0;
    for (int step = 0; step < step_num; step++) {
        for (int j = 0; j < positions.size(); j++) {
            std::vector<double> new_positions(positions.size());
            for (int i = 0; i < positions.size(); i++) {
                if (i == j) {
                    new_positions[i] = positions[i] + normal_step(gen);
                } else {
                    new_positions[i] = positions[i];
                }
            }
            if (step < int(skip_initial)) {continue;}
            std::vector<double> alpha_pos_old = std::vector<double>({alpha});
            alpha_pos_old.insert(alpha_pos_old.end(), positions.begin(), positions.end()); //TODO ugly :(
            std::vector<double> alpha_pos_new = std::vector<double>({alpha});
            alpha_pos_new.insert(alpha_pos_new.end(), new_positions.begin(), new_positions.end()); //TODO make this a fn
            double old_density = pow(trial_wavefunction(alpha_pos_old),2);
            double new_density = pow(trial_wavefunction(alpha_pos_new),2);
            double p = new_density / old_density;
            // case 1: higher likelihood: always keep
            if (p >= 1) {
                double E = local_energy(alpha_pos_new);
                current_energy += E / actual_steps; // moving average
                energies[energy_index] = E;
                positions = new_positions;
                // case 2: lower likelihood: only keep with probability p
            } else {
                double x = zero_one_uniform(gen);
                if (x < p) {
                    double E = local_energy(alpha_pos_new);
                    current_energy += E / actual_steps;
                    energies[energy_index] = E;
                    positions = new_positions;
                } else {
                    double E = local_energy(alpha_pos_old);
                    current_energy += E / actual_steps;
                    energies[energy_index] = E;
                }
            }
        energy_index += 1;
        }
    }
    std::vector<double> sq_diff(actual_steps, 0);
    std::generate(sq_diff.begin(), sq_diff.end(), [&, i=0]() mutable {
        double E = energies[i];
        i++;
        return pow(E - current_energy, 2);
    });
    stddev_energy = sqrt(std::accumulate(sq_diff.begin(), sq_diff.end(), 0) / double(actual_steps));
    return current_energy;
}

void monte_carlo::reset_positions() {
    positions = std::vector<double>(spatial_dim*particle_num, 0.0);
}

std::tuple<double, double, double> monte_carlo::get_best_trial() {
    if (!has_run) {
        throw std::runtime_error("Run the simulation before accessing results.");
    }
    double inf = std::numeric_limits<double>::max();
    double best_alpha = inf;
    double best_E = inf;
    double best_std = inf;
    for (int i = 0; i < trial_num; i++) {
        double trial_alpha = std::get<0>(alpha_energy_std[i]);
        double trial_energy = std::get<1>(alpha_energy_std[i]);
        double trial_std = std::get<2>(alpha_energy_std[i]);
        if (trial_energy < best_E) {
            best_E = trial_energy;
            best_alpha = trial_alpha;
            best_std = trial_std;
        }
    }
    return std::tuple<double, double, double>({best_alpha, best_E, best_std});
}

double monte_carlo::golden_split(double alpha1, double alpha2) {
    return alpha1 + 0.6180339887498949 * (alpha2 - alpha1);
}

void monte_carlo::save_trial() {
    std::ofstream file;
    file.open(out_path + "/trial_alpha_" + std::to_string(alpha) + ".txt");

    //get all digts
    for (int i = 0; i < energies.size() / 1000; i++) {
        int j = i * 1000;
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
             << std::scientific << energies[j] << std::endl;
    }

    file.close();
}

void monte_carlo::save_summary() {
    std::ofstream file;
    file.open(out_path + "/summary.txt");

    //get all digts
    for (int i = 0; i < trial_num; i++) {
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
             << std::scientific << std::get<0>(alpha_energy_std[i]) << ","
             << std::scientific << std::get<1>(alpha_energy_std[i]) << ","
             << std::scientific << std::get<2>(alpha_energy_std[i]) << std::endl;
    }

    file.close();
}

