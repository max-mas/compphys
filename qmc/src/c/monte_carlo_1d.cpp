//
// Created by max on 4/11/24.
//

#include "monte_carlo_1d.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

monte_carlo_1d::monte_carlo_1d(const unsigned int particleNum,
                               unsigned int trialNum,
                               unsigned int stepNum,
                               const std::function<double(std::vector<double>)> &trialWavefunction,
                               const std::function<double(std::vector<double>)> &localEnergy,
                               const std::pair<double, double> & alphaRange,
                               double step_stddev):
    trial_num(trialNum),
    particle_num(particleNum),
    step_num(stepNum),
    trial_wavefunction(trialWavefunction),
    local_energy(localEnergy),
    alpha_range(alphaRange)
{
    // init rng
    normal_step = std::normal_distribution<double>(0.0, step_stddev);
    // randomly init positions
    positions = std::vector<double>(particle_num, 0.0);
    std::generate(positions.begin(), positions.end(), [&](){return normal_step(gen);});
    // init energy(alpha) vector with zeros
    alpha_energy = std::vector<std::pair<double, double>>(trial_num, {0, 0});

    optimised_alpha_range = alpha_range;
}

double monte_carlo_1d::select_next_alpha() {
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
        alpha_x = golden_split(alpha_range.first, alpha_energy[2].first);
        return alpha_x;
    }

    if (alpha_energy[trial-1].second < alpha_energy[trial-2].second) {
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

void monte_carlo_1d::run() {
    if (has_run) {
        throw std::runtime_error("To run a new set of trials, create a new object!");
    }
    for (trial = 0; trial < trial_num; trial++) { //trial is a class member!
        alpha = select_next_alpha();
        std::cout << "Trial " << trial << ", doing " << step_num << " steps. alpha = " << alpha << std::flush;
        double energy = do_trial();
        std::cout << "; E = " << energy << std::endl;
        alpha_energy[trial] = {alpha, energy};
    }
    has_run = true;
    std::pair<double, double> best_alpha_E = get_best_trial();
    std::cout << "Best trial: alpha = " << best_alpha_E.first << ", E = " << best_alpha_E.second << std::endl;
    //std::cout   << "Highest confidence alpha range: [" << optimised_alpha_range.first << ", "
    //            << optimised_alpha_range.second << "]" << std::endl;
}

double monte_carlo_1d::do_trial() {
    reset_positions();
    current_energy = 0;
    std::uniform_real_distribution<double> zero_one_uniform(0, 1);
    unsigned int skip_initial = int(step_num) / 10; //TODO wasteful?
    unsigned int actual_steps = step_num - skip_initial;

    for (int step = 0; step < step_num; step++) {
        if (step < int(skip_initial)) {continue;}
        for (int j = 0; j < positions.size(); j++) {
            std::vector<double> new_positions(positions.size());
            for (int i = 0; i < positions.size(); i++) {
                if (i == j) {
                    new_positions[i] = positions[i] + normal_step(gen);
                } else {
                    new_positions[i] = positions[i];
                }
            }
            std::vector<double> alpha_pos_old = std::vector<double>({alpha});
            alpha_pos_old.insert(alpha_pos_old.end(), positions.begin(), positions.end()); //TODO ugly :(
            std::vector<double> alpha_pos_new = std::vector<double>({alpha});
            alpha_pos_new.insert(alpha_pos_new.end(), new_positions.begin(), new_positions.end()); //TODO make this a fn
            double old_density = pow(trial_wavefunction(alpha_pos_old),2);
            double new_density = pow(trial_wavefunction(alpha_pos_new),2);
            double p = new_density / old_density;
            // case 1: higher likelihood: always keep
            if (p >= 1) {
                current_energy += local_energy(alpha_pos_new) / actual_steps; // moving average
                positions = new_positions;
                // case 2: lower likelihood: only keep with probability p
            } else {
                double x = zero_one_uniform(gen);
                if (x < p) {
                    current_energy += local_energy(alpha_pos_new) / actual_steps;
                    positions = new_positions;
                } else {
                    current_energy += local_energy(alpha_pos_old) / actual_steps;
                }
            }
        }



    }
    return current_energy;
}

void monte_carlo_1d::reset_positions() {
    positions = std::vector<double>(particle_num, 0.0);
}

std::pair<double, double> monte_carlo_1d::get_best_trial() {
    if (!has_run) {
        throw std::runtime_error("Run the simulation before accessing results.");
    }
    double inf = std::numeric_limits<double>::max();
    double best_alpha = inf;
    double best_E = inf;
    for (int i = 0; i < trial_num; i++) {
        double trial_alpha = alpha_energy[i].first;
        double trial_energy = alpha_energy[i].second;
        if (trial_energy < best_E) {
            best_E = trial_energy;
            best_alpha = trial_alpha;
        }
    }
    return std::pair<double, double>({best_alpha, best_E});
}

double monte_carlo_1d::golden_split(double alpha1, double alpha2) {
    return alpha1 + 0.6180339887498949 * (alpha2 - alpha1);
}

