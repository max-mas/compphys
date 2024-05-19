/**
 * @file atom.h
 * @author Max Maschke (m.maschke@tu-bs.de)
 * @brief Templated class that implements a self consistent mean field approach for atoms.
 * @date 2024-05-07
 *
 * @copyright Copyright (c) 2024 Max Maschke
 *
 */

// STD imports
#include <functional>
#include <vector>
#include <map>
#include <tuple>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

// Eigen imports
#include <Eigen/Dense>

// Boost imports
#include <boost/math/quadrature/exp_sinh.hpp>

// User imports
#include "spherical_seq.h"
#include "collocation.h"

#ifndef PERIODICTAB_ATOM_H
#define PERIODICTAB_ATOM_H

/**
 * Approximates the electronic structure of atoms using a self-consistent mean field approach.
 * Solution of Schrödinger's and Poisson's equation is handled by external classes spherical_seq and collocation,
 * both based on algorithms using B-splines, implemented in the b_splines class.
 */
class atom {
public:
    int Z; // Nuclear charge number
    int N_e; // Number of electrons
    int max_it_self_consistency; // Maximum number of iterations
    int num_knots_seq; // Number of physical knots for the Schrödinger solver
    int num_knots_pot; // Number of physical knots for the Poisson solver
    int l_max = 3; // Maximum angular momentum quantum number. No electrons beyond f orbitals in nature!
    double r_max; // Maximum radius beyond which Psi = 0 is enforced
    double tol; // conv tolerance

    Eigen::MatrixXi occupation_matrix; // Matrix containing the occupation of all states found
    Eigen::MatrixXi prev_occupation_matrix;
    Eigen::MatrixXi orig_occupation_matrix = Eigen::MatrixXi::Zero(0, 0);

    // Key-value container, keys = energies (sorted in ascending order), values = (n, l) quantum numbers.
    // Only contains states with negative energies, i.e. bound states.
    std::map<double, std::pair<int, int>> bound_states;

    /**
     * Direct electronic interaction potential derived from the solution of
     * Poisson's equation for the previous ground state.
     * @param r Radial coordinate
     * @return V_ee_dir (r)
     */
    double electrostatic_potential(double r);

    /**
     * Direct electronic interaction potential derived from the solution of
     * Poisson's equation for the second-to-last ground state.
     * @param r Radial coordinate
     * @return V_ee_dir_prev (r)
     */
    double previous_electrostatic_potential(double r);

    /**
     * Electronic exchange interaction potential, using a free electron gas approximation,
     * involving the ground state electronic charge density obtained from the previous SEQ solution:
     * V_ee_ex (r) = -3 (3/(8 Pi) rho(r)) ^ 1/3
     * @param r Radial coordinate
     * @return V_ee_ex (r)
     */
    double exchange_potential(double r);

    /**
     * Electronic exchange interaction potential, using a free electron gas approximation,
     * involving the ground state electronic charge density obtained from the second-to-last SEQ solution:
     * V_ee_ex (r) = -3 (3/(8 Pi) rho(r)) ^ 1/3
     * @param r Radial coordinate
     * @return V_ee_ex_prev (r)
     */
    double previous_exchange_potential(double r);

    /**
     * Complete electronic interaction potential:
     * V_ee (r) = V_ee_dir (r) + V_ee_ex (r)
     * @param r Radial coordinate
     * @return V_ee (r)
     */
    double many_body_potential(double r);

    /**
     * Complete electronic interaction potential, using the second-to-last solution:
     * V_ee (r) = V_ee_dir (r) + V_ee_ex (r)
     * @param r Radial coordinate
     * @return V_ee_prev (r)
     */
    double previous_many_body_potential(double r);

    /**
     * Complete electronic interaction potential, mixing the current and previous solutions:
     * V_ee_mix (r) = (1 - eta) (V_ee_dir (r) + V_ee_ex (r)) + eta (V_ee_dir_prev (r) + V_ee_ex_prev (r))
     * @param r Radial coordinate
     * @return V_ee_mix (r)
     */
    double mixed_many_body_potential(double r);

    /**
     * Nuclear coulomb potential.
     * V_c (r) = -Z/r
     * @param r Radial coordinate
     * @return V_c (r)
     */
    double nuclear_potential(double r);

    /**
     * Full mean-field potential, including the nucleus and many-body interaction.
     * @param r Radial coordinate
     * @return V(r)
     */
    double mean_field_potential(double r);

    /**
     * Full mean-field potential, mixing the previous and second-to-last solutions.
     * V_mix (r) = (1 - eta) V (r) - eta V_prev (r)
     * @param r Radial coordinate
     * @return V_mix (r)
     */
    double mixed_mean_field_potential(double r);

    /**
     * Returns zero.
     * @param r Radial coordinate.
     * @return 0
     */
    double zero_function(double r);

    /**
     * Right hand side of Poisson's equation used to determine the interaction potential V_ee.
     * Poisson's equation in spherical symmetry is
     * d^2/dr^2 phi(r) = - 4 Pi r rho(r) / (4 Pi eps_0)
     * where V(r) = phi(r) / r.
     * @param r Radial coordinate.
     * @return Poisson eqn rhs.
     */
    double colloc_rhs(double r);

    double prev_colloc_rhs(double r);

    /**
     * Computs the total atomic ground satte energy from the occupied orbitals, subtracting self-interaction.
     * @return E_tot
     */
    double total_energy();

    /**
     * Default constructor for the atom class.
     */
    atom() = default;

    /**
     * Atom constructor.
     * @param chargeNumber Nuclear charge number.
     * @param electronNumber Number of electrons.
     * @param maxIt Maximum number of self-consistency iterations.
     * @param rMax Maximum radius beyond which Psi, rho = 0 are enforced.
     * @param numKnotsSeq Number of knots / points for the Schrödinger eqn solution.
     * @param numKnotsPot Number of knots / points for the Poisson's eqn solution.
     * @param occupationInit
     * @param Eta
     */
    atom(int chargeNumber, int electronNumber, int maxIt,
         double rMax, int numKnotsSeq, int numKnotsPot,
         double Eta = 0.4, double tolerance = 5e-4, 
         Eigen::MatrixXi occupationInit = Eigen::MatrixXi::Zero(1, 1));

    /**
     * Begin the iteration.
     * @throws std::runtime_error if run() has been called before.
     */
    void run();

    std::vector<spherical_seq> seq_solvers; // Vector of spherical_seq objects for different l.
    std::vector<spherical_seq> previous_seq_solvers;
    collocation potential_solver; // collocation object for solving Poisson's equation.
    collocation previous_potential_solver; // Previous collocation object, used for the mixed potential.

    /**
     * Saves the orbital occupation, the orbital energies and the total energy to a file at path.
     * @param path Path to .txt file.
     */
    void save_summary(const std::string& path);

    /**
     * Saves the direct and exchange interaction potentials to files in a directory at path.
     * @param path Path to a directory.
     */
    void save_potentials(const std::string& path);

    /**
     * Saves the electron charge density to a file at path.
     * @param path Path to a .txt file.
     */
    void save_rho(const std::string& path);

    /**
     * Generates an Eigen::Vector of exponentially spaced grid points from 0 to R.
     * @param n Number of points.
     * @param R Endpoint.
     * @return Eigen column vector containing the points.
     */
    Eigen::VectorXd generate_knots(int n, double R);

private:
    double eta;

    double curr_erg; // Current 1s state energy, used for convergence checking.
    double prev_erg = 0.0; // Previous 1s state energy, used for convergence checking.

    Eigen::VectorXd knots_seq; // Physical knot sequence for the SEQ solver.
    Eigen::VectorXd knots_pot; // Physical knot sequence for the Poisson solver.

    /**
     * Performs one iteration step.
     * @param i count
     */
    void iteration(int i);

    /**
     * Prints the integral of the electronic charge density, which ought to equal the number of electrons.
     */
    void check_rho();

    /**
     * Electronic charge density, computed from the occupied orbitals.
     * @param r Radial coordinate.
     * @returns rho(r)
     */
    double rho(double r);

    double prev_rho(double r);


    /**
     * Assigns the electrons to the orbitals found by solving the SEQ.
     * @returns Integer matrix encoding the occupation of the states.
     */
    Eigen::MatrixXi determine_occupation();

    bool init_finished = false; // Flag tht tracks if initialised
    bool second_it_finished = false; // Flag that tracks if first iteration complete
    bool ran = false; // Flag that tracks if iteration complete

    bool assign_orbitals_auto = true;

    // Maps l quantum numbers to orbital names s, p, d, f, g, h, j
    std::map<int, std::string> orbital_names =
        std::map<int, std::string>({{0, "s"}, {1, "p"}, {2, "d"}, {3, "f"}, {4, "g"},
                                             {5, "h"}, {6, "j"}});
};


#endif //PERIODICTAB_ATOM_H
