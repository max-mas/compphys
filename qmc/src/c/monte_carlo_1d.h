//
// Created by max on 4/11/24.
//

#ifndef QMC_MONTE_CARLO_1D_H
#define QMC_MONTE_CARLO_1D_H

#include <functional>
#include <tuple>
#include <complex>
#include <vector>
#include <random>

class monte_carlo_1d {
public:
    const unsigned int particle_num;
    const unsigned int trial_num;
    const unsigned int step_num;
    const std::pair<double, double> alpha_range;
    const std::function<double (std::vector<double>)> trial_wavefunction;
    const std::function<double (std::vector<double>)> local_energy;

    monte_carlo_1d(unsigned int particleNum,
                   unsigned int trialNum,
                   unsigned int stepNum,
                   const std::function<double(std::vector<double>)> &trialWavefunction,
                   const std::function<double(std::vector<double>)> &localEnergy,
                   const std::pair<double, double> & alphaRange,
                   double step_stddev);

    void run(); //TODO implement convergence testing / early stopping
    std::pair<double, double> get_best_trial();


private:
    //random numbers
    std::random_device rd{}; //use hardware rng to seed mersenne twister
    std::mt19937_64 gen{rd()}; //pseudo rng for normal distribution
    std::normal_distribution<double> normal_step;

    std::vector<double> positions;

    double select_next_alpha(); // TODO now only random samples
    double do_trial();
    void reset_positions();
    static double golden_split(double alpha1, double alpha2);

    bool has_run = false;
    int trial = 0;
    double alpha;
    double current_energy;
    std::vector<std::pair<double, double>> alpha_energy;
    std::pair<double, double> optimised_alpha_range;

};


#endif //QMC_MONTE_CARLO_1D_H
