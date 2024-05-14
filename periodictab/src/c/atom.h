/**
 * @file spherical_seq.h
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
 * @tparam numeric_type Real numeric type like double, float.
 */
template <typename numeric_type>
class atom {
public:
    unsigned int Z; // Nuclear charge number
    unsigned int N_e; // Number of electrons
    unsigned int max_it_self_consistency; // Maximum number of iterations
    unsigned int num_knots_seq; // Number of physical knots for the Schrödinger solver
    unsigned int num_knots_pot; // Number of physical knots for the Poisson solver
    unsigned int l_max = 3; // Maximum angular momentum quantum number. No electrons beyond f orbitals in nature!
    numeric_type r_max; // Maximum radius beyond which Psi = 0 is enforced

    Eigen::MatrixXi occupation_matrix; // Matrix containing the occupation of all states found
    Eigen::MatrixXi prev_occupation_matrix;

    // Key-value container, keys = energies (sorted in ascending order), values = (n, l) quantum numbers.
    // Only contains states with negative energies, i.e. bound states.
    std::map<numeric_type, std::pair<unsigned int, unsigned int>> bound_states;

    /**
     * Direct electronic interaction potential derived from the solution of
     * Poisson's equation for the previous ground state.
     * @param r Radial coordinate
     * @return V_ee_dir (r)
     */
    numeric_type electrostatic_potential(numeric_type r);

    /**
     * Direct electronic interaction potential derived from the solution of
     * Poisson's equation for the second-to-last ground state.
     * @param r Radial coordinate
     * @return V_ee_dir_prev (r)
     */
    numeric_type previous_electrostatic_potential(numeric_type r);

    /**
     * Electronic exchange interaction potential, using a free electron gas approximation,
     * involving the ground state electronic charge density obtained from the previous SEQ solution:
     * V_ee_ex (r) = -3 (3/(8 Pi) rho(r)) ^ 1/3
     * @param r Radial coordinate
     * @return V_ee_ex (r)
     */
    numeric_type exchange_potential(numeric_type r);

    /**
     * Electronic exchange interaction potential, using a free electron gas approximation,
     * involving the ground state electronic charge density obtained from the second-to-last SEQ solution:
     * V_ee_ex (r) = -3 (3/(8 Pi) rho(r)) ^ 1/3
     * @param r Radial coordinate
     * @return V_ee_ex_prev (r)
     */
    numeric_type previous_exchange_potential(numeric_type r);

    /**
     * Complete electronic interaction potential:
     * V_ee (r) = V_ee_dir (r) + V_ee_ex (r)
     * @param r Radial coordinate
     * @return V_ee (r)
     */
    numeric_type many_body_potential(numeric_type r);

    /**
     * Complete electronic interaction potential, using the second-to-last solution:
     * V_ee (r) = V_ee_dir (r) + V_ee_ex (r)
     * @param r Radial coordinate
     * @return V_ee_prev (r)
     */
    numeric_type previous_many_body_potential(numeric_type r);

    /**
     * Complete electronic interaction potential, mixing the current and previous solutions:
     * V_ee_mix (r) = (1 - eta) (V_ee_dir (r) + V_ee_ex (r)) + eta (V_ee_dir_prev (r) + V_ee_ex_prev (r))
     * @param r Radial coordinate
     * @return V_ee_mix (r)
     */
    numeric_type mixed_many_body_potential(numeric_type r);

    /**
     * Nuclear coulomb potential.
     * V_c (r) = -Z/r
     * @param r Radial coordinate
     * @return V_c (r)
     */
    numeric_type nuclear_potential(numeric_type r);

    /**
     * Full mean-field potential, including the nucleus and many-body interaction.
     * @param r Radial coordinate
     * @return V(r)
     */
    numeric_type mean_field_potential(numeric_type r);

    /**
     * Full mean-field potential, mixing the previous and second-to-last solutions.
     * V_mix (r) = (1 - eta) V (r) - eta V_prev (r)
     * @param r Radial coordinate
     * @return V_mix (r)
     */
    numeric_type mixed_mean_field_potential(numeric_type r);

    /**
     * Returns zero.
     * @param r Radial coordinate.
     * @return 0
     */
    numeric_type zero_function(numeric_type r);

    /**
     * Right hand side of Poisson's equation used to determine the interaction potential V_ee.
     * Poisson's equation in spherical symmetry is
     * d^2/dr^2 phi(r) = - 4 Pi r rho(r) / (4 Pi eps_0)
     * where V(r) = phi(r) / r.
     * @param r Radial coordinate.
     * @return Poisson eqn rhs.
     */
    numeric_type colloc_rhs(numeric_type r);

    numeric_type prev_colloc_rhs(numeric_type r);

    /**
     * Computs the total atomic ground satte energy from the occupied orbitals, subtracting self-interaction.
     * @return E_tot
     */
    numeric_type total_energy();

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
     */
    atom(unsigned int chargeNumber, unsigned int electronNumber, unsigned int maxIt,
         numeric_type rMax, unsigned int numKnotsSeq, unsigned int numKnotsPot);

    /**
     * Begin the iteration.
     * @throws std::runtime_error if run() has been called before.
     */
    void run();

    std::vector<spherical_seq<numeric_type>> seq_solvers; // Vector of spherical_seq objects for different l.
    std::vector<spherical_seq<numeric_type>> previous_seq_solvers;
    collocation<numeric_type> potential_solver; // collocation object for solving Poisson's equation.
    collocation<numeric_type> previous_potential_solver; // Previous collocation object, used for the mixed potential.

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

private:
    numeric_type curr_erg; // Current 1s state energy, used for convergence checking.
    numeric_type prev_erg = 0.0; // Previous 1s state energy, used for convergence checking.

    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> knots_seq; // Physical knot sequence for the SEQ solver.
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> knots_pot; // Physical knot sequence for the Poisson solver.

    /**
     * Generates an Eigen::Vector of exponentially spaced grid points from 0 to R.
     * @param n Number of points.
     * @param R Endpoint.
     * @return Eigen column vector containing the points.
     */
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> generate_knots(unsigned int n, numeric_type R);

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
    numeric_type rho(numeric_type r);

    numeric_type prev_rho(numeric_type r);


    /**
     * Assigns the electrons to the orbitals found by solving the SEQ.
     * @returns Integer matrix encoding the occupation of the states.
     */
    Eigen::MatrixXi determine_occupation();

    bool init_finished = false; // Flag tht tracks if initialised
    bool second_it_finished = false; // Flag that tracks if first iteration complete
    bool ran = false; // Flag that tracks if iteration complete

    // Maps l quantum numbers to orbital names s, p, d, f, g, h, j
    std::map<unsigned int, std::string> orbital_names =
        std::map<unsigned int, std::string>({{0, "s"}, {1, "p"}, {2, "d"}, {3, "f"}, {4, "g"},
                                             {5, "h"}, {6, "j"}});
};


template<typename numeric_type>
atom<numeric_type>::atom(unsigned int chargeNumber, unsigned int electronNumber, unsigned int maxIt,
                         numeric_type rMax, unsigned int numKnotsSeq, unsigned int numKnotsPot):
Z(chargeNumber), N_e(electronNumber), max_it_self_consistency(maxIt), r_max(rMax),
num_knots_seq(numKnotsSeq), num_knots_pot(numKnotsPot) {
    // Generate knot sequences
    this->knots_seq = generate_knots(this->num_knots_seq, this->r_max);
    this->knots_pot = generate_knots(this->num_knots_pot, this->r_max);

    // Needed to prevent segfault
    this->seq_solvers = std::vector<spherical_seq<numeric_type>>();

    // Potentially needed to prevent segfault
    this->potential_solver = collocation<numeric_type>();

    // Solve SEQ for all l
    std::cout << "Find atomic structure for Z = " << this->Z << " and " << this->N_e << " electrons." << std::endl;
    std::cout << "Initial SEQ solution with V = 0." << std::endl;
    for (int l = 0; l <= this->l_max; l++) {
        std::cout << "l = " << l << ", " << std::flush;
        auto fn = [&](numeric_type r){return this->mean_field_potential(r);};
        this->seq_solvers.emplace_back(spherical_seq<numeric_type>(this->knots_seq, fn, l));
        this->seq_solvers[l].solve();
    }
    std::cout << std::endl;

    // Assign electrons
    this->occupation_matrix = this->determine_occupation();
    // Save for convergence checking
    this->curr_erg = this->total_energy();
    // Print total energy
    std::cout << "Total energy: " << this->total_energy() << std::endl;

    // Solve poisson eqn
    auto zero_fn = [](numeric_type r){return 0.0;};
    this->potential_solver = collocation<numeric_type>(zero_fn, zero_fn,
                                                       [&](numeric_type r){return this->colloc_rhs(r);},
                                                       this->knots_pot, {0, N_e});
    this->potential_solver.solve();
    std::cout << "Initialised rho solution..." << std::endl;
    // Sanity check
    this->check_rho();
    // Set init flag
    this->init_finished = true;
}

template<typename numeric_type>
void atom<numeric_type>::run() {
    // Throw a fit if run() has been called before
    if (this->ran) throw std::runtime_error("Computation has been run before. Create a new object.");

    // Tolerance for convergence check of 1s energy TODO what would a sensible value be? rn 4 decimals
    numeric_type tol = 5e-4;
    for (int i = 1; i <= this->max_it_self_consistency; i++) {
        std::cout << "Iteration " << i << ":" << std::endl;
        this->iteration(i); // perform an iteration
        numeric_type change_erg = abs(this->curr_erg - this->prev_erg);
        if (change_erg < tol) { // convergence check
            std::cout << "Tolerance of " << tol << " reached after " << i << " iterations. Stopping." << std::endl;
            this->ran = true; // flag
            return;
        }
    }
    // If not converged:
    std::cout << "Tolerance of " << tol << " not reached after " << this->max_it_self_consistency
        << " iterations. Stopping." << std::endl;

    this->ran = true; // flag
}

template<typename numeric_type>
void atom<numeric_type>::iteration(int i) {
    //TODO
    this->previous_seq_solvers = this->seq_solvers;
    this->prev_occupation_matrix = this->occupation_matrix;
    // Solve SEQ for all l
    for (int l = 0; l <= this->l_max; l++) {
        std::cout << "l = " << l << ", " << std::flush;
        this->seq_solvers[l] = spherical_seq<numeric_type>(this->knots_seq,
                                                [&](numeric_type r){return this->mixed_mean_field_potential(r);}, l);
        this->seq_solvers[l].solve();
    }
    std::cout << std::endl;

    // Assign electrons to orbitals
    this->occupation_matrix = this->determine_occupation();
    // For convergence checking
    numeric_type e = this->total_energy();
    this->prev_erg = this->curr_erg;
    this->curr_erg = e;
    // Print total energy
    std::cout << "Total energy: " << e << std::endl;

    // Solve Poisson's eqn
    this->previous_potential_solver = this->potential_solver;
    auto zero_fn = [](numeric_type r){return 0.0;};
    this->potential_solver = collocation<numeric_type>(zero_fn, zero_fn,
                                                       [&](numeric_type r){return this->colloc_rhs(r);},
                                                       this->knots_pot, {0, N_e});
    this->potential_solver.solve();
    std::cout << "Updated rho solution..." << std::endl;
    // Sanity check
    this->check_rho();
    if (i == 1) this->second_it_finished = true; // flag

}

// row index l, column index n
template<typename numeric_type>
Eigen::MatrixXi atom<numeric_type>::determine_occupation() {
    Eigen::MatrixXi retmat = Eigen::MatrixXi::Zero(this->l_max + 1, this->knots_seq.size());
    // key: E, value: l, n (n-1 here!)
    this->bound_states = std::map<numeric_type, std::pair<unsigned int, unsigned int>>();
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            numeric_type E = this->seq_solvers[l].energies(n);
            if (E < 0) {
                this->bound_states[E] = {l, n+l}; // this n corresponds to n - 1  !!!
            } else {
                break;
            }
        }
    }
    // assign electrons to minimise energy until all assigned
    unsigned int unassigned_electrons = this-> N_e;
    for (const auto& [E, qnums] : this->bound_states) {
        if (unassigned_electrons > 0) {
            unsigned int l = qnums.first;
            unsigned int n = qnums.second;
            unsigned int multiplicity = 2 * (2 * l + 1);

            for (int it = 0; it < multiplicity; it++) {
                if (unassigned_electrons > 0) {
                    retmat(l, n) += 1;
                    unassigned_electrons -= 1;
                } else break;
            }
        } else break;
    }
    // Print summary of occupancy and orbital energies
    std::cout << "Occupancy:\n";
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            if (retmat(l,n) > 0) std::cout << n+1 << this->orbital_names[l] << ": "
                << retmat(l,n) <<", E = " << this->seq_solvers[l].energies(n-l) << "\n"; // n - l !
        }
    }

    return retmat;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::total_energy() {
    numeric_type ret = 0.0;
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            unsigned int occ = this->occupation_matrix(l, n);
            // skip integration for unoccupied / invalid states
            if (occ < 1) continue; else if (l > n) break;

            // V expectation value integral
            boost::math::quadrature::exp_sinh<numeric_type> integrator;
            auto f = [&](numeric_type r)
                    {return pow(r * this->seq_solvers[l].solution_n(n-l, r), 2) * this->mixed_many_body_potential(r);};
            numeric_type V_int = integrator.integrate(f);
            //
            ret += occ * (this->seq_solvers[l].energies(n-l) - 0.5 * V_int); // n - l !
        }
    }
    return ret;
}

template<typename numeric_type>
void atom<numeric_type>::check_rho() {
    // Integrate rho and print to see if result equals N_e
    boost::math::quadrature::exp_sinh<numeric_type> integrator; // domain [0, inf)
    auto f = [&](numeric_type r) {  if (r<=0) r = std::numeric_limits<numeric_type>::min();
                                    return pow(r, 2) * this->rho(r) ;};
    numeric_type N = 4.0 * M_PI *  integrator.integrate(f);
    std::cout << "Numerical N_e: " << N << ", expected: " << this->N_e << std::endl;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::rho(numeric_type r) {
    if (r <= 0) r = std::numeric_limits<numeric_type>::epsilon();
    return this->colloc_rhs(r) / (-r * 4.0 * M_PI) ;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::prev_rho(numeric_type r) {
    if (r <= 0) r = std::numeric_limits<numeric_type>::epsilon();
    return this->prev_colloc_rhs(r) / (-r * 4.0 * M_PI) ;
}

template<typename numeric_type>
Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> atom<numeric_type>::generate_knots(unsigned int n, numeric_type R) {
    //return Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>::LinSpaced(n, 0.0, R);
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> points =
            Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>::Zero(n);
    numeric_type exponent = -6;
    numeric_type end_exponent = log10(R);
    numeric_type exponent_step = (end_exponent - exponent) / (n - 2);
    for (int i = 1; i < points.size(); i++) {
        points(i) = pow(10, exponent);
        exponent += exponent_step;
    }

    return points;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::electrostatic_potential(numeric_type r) {
    if (r <= 0.0) r = std::numeric_limits<numeric_type>::epsilon();
    return (this->init_finished) ? this->potential_solver.solution(r)/r : 0.0;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::exchange_potential(numeric_type r) {
    if (r <= 0.0) r = std::numeric_limits<numeric_type>::epsilon();
    // 4 pi epsilon = e = 1
    return (this->init_finished) ?
        -3.0 * pow(3.0 * abs(this->rho(r)) / (8.0 * numeric_type(M_PI)), 1.0/3.0) : 0.0;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::many_body_potential(numeric_type r) {
    // e = 1
    return (this->N_e > 1) ? 1.0 * this->electrostatic_potential(r) + 1.0 * this->exchange_potential(r) : 0.0;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::previous_electrostatic_potential(numeric_type r) {
    if (r <= 0.0) r = std::numeric_limits<numeric_type>::epsilon();
    return (this->second_it_finished) ? this->previous_potential_solver.solution(r)/r : 0.0;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::previous_exchange_potential(numeric_type r) {
    if (r <= 0.0) r = std::numeric_limits<numeric_type>::epsilon();
    // 4 pi epsilon = e = 1
    return (this->second_it_finished) ?
        -3.0 * pow(3.0 * abs(this->prev_rho(r)) / (8.0 * numeric_type(M_PI)), 1.0/3.0) : 0.0;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::previous_many_body_potential(numeric_type r) {
    // e = 1
    return (this->N_e > 1) ?
        1.0 * this->previous_electrostatic_potential(r) + 1.0 * this->previous_exchange_potential(r) : 0.0;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::nuclear_potential(numeric_type r) {
    return -numeric_type(this->Z) / r;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::mean_field_potential(numeric_type r) {
    return this->nuclear_potential(r) + this->many_body_potential(r);
}

template<typename numeric_type>
numeric_type atom<numeric_type>::mixed_mean_field_potential(numeric_type r) {
    numeric_type eta = 0.4;
    return this->nuclear_potential(r) + (1 - eta) * this->many_body_potential(r)
                                      +      eta  * this->previous_many_body_potential(r);
}

template<typename numeric_type>
numeric_type atom<numeric_type>::mixed_many_body_potential(numeric_type r) {
    numeric_type eta = 0.4;
    return (1 - eta) * this->many_body_potential(r) + eta * this->previous_many_body_potential(r);
}

template<typename numeric_type>
numeric_type atom<numeric_type>::zero_function(numeric_type r) {
    return 0.0;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::colloc_rhs(numeric_type r) {
    numeric_type ret = 0.0;
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            int occ = this->occupation_matrix(l, n);
            ret += (occ > 0) ? 1.0 * occ * pow(this->seq_solvers[l].solution_n(n-l, r), 2) : 0.0; // - <=> e??
        }
    }
    return -r * ret;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::prev_colloc_rhs(numeric_type r) {
    numeric_type ret = 0.0;
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            int occ = this->prev_occupation_matrix(l, n);
            ret += (occ > 0) ? 1.0 * occ * pow(this->previous_seq_solvers[l].solution_n(n-l, r), 2) : 0.0; // - <=> e??
        }
    }
    return -r * ret;
}

template<typename numeric_type>
void atom<numeric_type>::save_summary(const std::string& path) {
    std::ofstream file;
    file.open(path);
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            if (this->occupation_matrix(l,n) > 0) {
                file << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                     << n + 1 << this->orbital_names[l] << ": "
                     << this->occupation_matrix(l, n) << ", " << std::scientific << this->seq_solvers[l].energies(n - l)
                     << std::endl; // n - l !
            }
        }
    }
    file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific
         << this->total_energy() << std::endl;
    // don't forget to clean up :)
    file.close();
}

template<typename numeric_type>
void atom<numeric_type>::save_potentials(const std::string& path) {
    std::ofstream file1;
    std::ofstream file2;

    unsigned int num_pts = 3000;
    auto points = Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>::LinSpaced(num_pts, 0.0, this->r_max);

    file1.open(path + "direct_ne_" + std::to_string(this->N_e) + ".txt");
    file2.open(path + "exchange_ne_" + std::to_string(this->N_e) + ".txt");
    for (auto x : points) {
        file1 << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
              << std::scientific << x << "," << std::scientific << this->electrostatic_potential(x) << std::endl;

        file2 << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
              << std::scientific << x << "," << std::scientific << this->exchange_potential(x) << std::endl;
    }
    // don't forget to clean up :)
    file1.close();
    file2.close();
}

template<typename numeric_type>
void atom<numeric_type>::save_rho(const std::string& path) {
    std::ofstream file;

    unsigned int num_pts = 3000;
    auto points = Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>::LinSpaced(num_pts, 0.0, this->r_max);

    file.open(path);
    for (auto x : points) {
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
              << std::scientific << x << "," << std::scientific << this->rho(x) << std::endl;
    }
    // don't forget to clean up :)
    file.close();
}

#endif //PERIODICTAB_ATOM_H
