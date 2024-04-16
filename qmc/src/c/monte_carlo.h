//
// Created by max on 4/11/24.
//

#ifndef QMC_MONTE_CARLO_H
#define QMC_MONTE_CARLO_H

#include <functional>
#include <tuple>
#include <complex>
#include <vector>
#include <random>
#include <string>

/**
 * This class implements the metropolis monte carlo algorithm.
 * It ought to work for a general number of particles in a general number of dimensions,
 * but only accepts trial wave functions with exactly one optimisation parameter alpha!
 * The trial wf and the local energy function have to be supplied as explicit function references.
 */
class monte_carlo {
public:
    const unsigned int particle_num; // Number of particles
    const unsigned int spatial_dim; // Number of spatial dimensions
    const unsigned int trial_num; // Number of trials, i.e. alpha values to check
    const unsigned int step_num; // Number of steps per trial (not exact, see implementation of do_trial() )
    const std::pair<double, double> alpha_range; // Interval to search optimal alpha in
    const std::function<double (std::vector<double>)> trial_wavefunction; // Vector parameter s.t. vec[0] = alpha,
    const std::function<double (std::vector<double>)> local_energy;       // rest are positions of the particles
    bool save_trials; // whether to save
    bool spaced_alphas;

    /**
     * Constructor.
     * @param particleNum // see above
     * @param spatialDim
     * @param trialNum
     * @param stepNum
     * @param trialWavefunction
     * @param localEnergy
     * @param alphaRange
     * @param step_stddev // Std dev of normal distr. to pick offsets to vary position from
     * @param saveTrials  // Whether to save samples of the trials and a summary to a directory. Default: False
     * @param outPath  // Directory to save to, default: empty string
     * @param spacedAlphas // for plotting
     */
    monte_carlo(unsigned int particleNum,
                   unsigned int spatialDim,
                   unsigned int trialNum,
                   unsigned int stepNum,
                   const std::function<double(std::vector<double>)> &trialWavefunction,
                   const std::function<double(std::vector<double>)> &localEnergy,
                   const std::pair<double, double> & alphaRange,
                   double step_stddev,
                   bool saveTrials = false,
                   std::string  outPath = "",
                   bool spacedAlphas = false);

    /*
     * Call to start the trials.
     */
    void run(); //TODO implement convergence testing / early stopping
    std::tuple<double, double, double> get_best_trial(); // Call to get the best alpha, E, std after running.


private:
    //random numbers
    std::random_device rd{}; //use hardware rng to seed mersenne twister
    std::mt19937_64 gen{rd()}; //pseudo rng for normal distribution
    std::normal_distribution<double> normal_step; // normal dist for pos offsets

    std::vector<double> positions; // position vector for iteration

    std::string out_path; // path to output dir

    /**
     * Golden search implementation.
     * @return next alpha to check
     */
    double select_next_alpha();

    /**
     * For plotting: linspaced alphas
     * @return next alpha to check
     */
    double next_alpha_spaced();

    /**
     * Do next trial.
     * @return energy of trial
     */
    double do_trial();

    /**
     * Reset positions at start of trial.
     */
    void reset_positions();

    /**
     * Helper function that splits interval acc. to golden ratio.
     * @param alpha1 left boundary
     * @param alpha2 right boundary
     * @return split
     */
    static double golden_split(double alpha1, double alpha2);

    /**
     * Save sampled trial to file. Samples every 1000th step to reduce overhead and storage consumption.
     */
    void save_trial();

    /**
     * Save summary of full run.
     */
    void save_summary();

    bool has_run = false; // if set, refuse to rerun
    int trial = 0; // keep track of current trial
    double alpha; // current alpha
    double current_energy; // current energy
    std::vector<double> energies; // all current trial energies
    double stddev_energy; // trial stddev
    std::vector<std::tuple<double, double, double>> alpha_energy_std; // stores all trial results
    std::pair<double, double> optimised_alpha_range; // golden search interval
};


#endif //QMC_MONTE_CARLO_H
