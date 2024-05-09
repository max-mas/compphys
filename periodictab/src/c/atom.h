//
// Created by max on 5/7/24.
//
#include <functional>
#include <vector>
#include <map>
#include <tuple>
#include <cmath>
#include <string>
#include <iostream>

#include <Eigen/Dense>

#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>

#include "spherical_seq.h"
#include "collocation.h"

//TODO rm
#include <fenv.h>

#ifndef PERIODICTAB_ATOM_H
#define PERIODICTAB_ATOM_H

template <typename numeric_type>
class atom {
public:
    unsigned int Z;
    unsigned int N_e;
    unsigned int max_it_self_consistency;
    unsigned int num_knots_seq;
    unsigned int num_knots_pot;
    unsigned int l_max;
    numeric_type r_max;

    Eigen::MatrixXi occupation_matrix;

    std::map<numeric_type, std::pair<unsigned int, unsigned int>> bound_states;

    numeric_type electrostatic_potential(numeric_type r);
    numeric_type previous_electrostatic_potential(numeric_type r);

    numeric_type exchange_potential(numeric_type r);
    numeric_type previous_exchange_potential(numeric_type r);

    numeric_type many_body_potential(numeric_type r);
    numeric_type previous_many_body_potential(numeric_type r);

    numeric_type nuclear_potential(numeric_type r);

    numeric_type mean_field_potential(numeric_type r);
    numeric_type mixed_mean_field_potential(numeric_type r);

    static numeric_type zero_function(numeric_type r);

    numeric_type colloc_rhs(numeric_type r);

    numeric_type total_energy();

    atom() = default;

    atom(unsigned int chargeNumber, unsigned int electronNumber, unsigned int maxIt,
         numeric_type rMax, unsigned int numKnotsSeq, unsigned int numKnotsPot);

    void run();

private:
    std::vector<spherical_seq<numeric_type>> seq_solvers;
    collocation<numeric_type> potential_solver;
    collocation<numeric_type> previous_potential_solver;

    numeric_type curr_1s_erg;
    numeric_type prev_1s_erg = 0.0;

    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> knots_seq;
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> knots_pot;

    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> generate_knots(unsigned int n, numeric_type R);

    void iteration(int i);

    void check_rho();

    numeric_type rho(numeric_type r);

    Eigen::MatrixXi determine_occupation();

    bool init_finished = false;
    bool second_it_finished = false;

    std::map<unsigned int, std::string> orbital_names =
        std::map<unsigned int, std::string>({{0, "s"}, {1, "p"}, {2, "d"}, {3, "f"}, {4, "g"},
                                             {5, "h"}, {6, "i"}});
};


template<typename numeric_type>
atom<numeric_type>::atom(unsigned int chargeNumber, unsigned int electronNumber, unsigned int maxIt,
                         numeric_type rMax, unsigned int numKnotsSeq, unsigned int numKnotsPot):
Z(chargeNumber), N_e(electronNumber), max_it_self_consistency(maxIt), r_max(rMax),
num_knots_seq(numKnotsSeq), num_knots_pot(numKnotsPot) {
    this->l_max = 6; // TODO eh
    this->knots_seq = generate_knots(this->num_knots_seq, this->r_max);
    this->knots_pot = generate_knots(this->num_knots_pot, this->r_max);
    this->seq_solvers.reserve(this->l_max+1);

    std::cout << "Find atomic structure for Z = " << this->Z << " and " << this->N_e << " electrons." << std::endl;
    std::cout << "Initial SEQ solution with V = 0." << std::endl;
    for (int l = 0; l <= this->l_max; l++) {
        std::cout << "l = " << l << ", " << std::flush;
        this->seq_solvers[l] = spherical_seq<numeric_type>(this->knots_seq,
                                                   [&](numeric_type r){return this->mean_field_potential(r);}, l);
        this->seq_solvers[l].solve();
    }
    std::cout << std::endl;

    this->occupation_matrix = this->determine_occupation();
    this->curr_1s_erg = this->seq_solvers[0].energies(0);
    //std::cout << occupation_matrix << std::endl;
    std::cout << "Total energy: " << this->total_energy() << std::endl;

    this->potential_solver = collocation<numeric_type>(this->zero_function, this->zero_function,
                                                       [&](numeric_type r){return this->colloc_rhs(r);},
                                                       this->knots_pot, {0, N_e}); // TODO BC correct??

    this->potential_solver.solve();
    std::cout << "Initialised rho solution..." << std::endl;
    this->check_rho();

    this->potential_solver.save_solution(0, this->r_max, 1000, "../results/test.txt"); //TODO rm

    this->init_finished = true;
}

template<typename numeric_type>
void atom<numeric_type>::run() {
    numeric_type tol = 1e-6;
    for (int i = 1; i <= this->max_it_self_consistency; i++) {
        std::cout << "Iteration " << i << ":" << std::endl;
        this->iteration(i);
        numeric_type change_1s_erg = abs(this->curr_1s_erg - this->prev_1s_erg);
        if (change_1s_erg < tol) {
            std::cout << "Tolerance of " << tol << " reached after " << i << " iterations. Stopping." << std::endl;
            return;
        }
    }
    std::cout << "Tolerance of " << tol << " not reached after " << this->max_it_self_consistency
        << " iterations. Stopping." << std::endl;
}

template<typename numeric_type>
void atom<numeric_type>::iteration(int i) {
    //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    for (int l = 0; l <= this->l_max; l++) {
        std::cout << "l = " << l << ", ";
        this->seq_solvers[l] = spherical_seq<numeric_type>(this->knots_seq,
                                                [&](numeric_type r){return this->mixed_mean_field_potential(r);}, l); //TODO mixed
        this->seq_solvers[l].solve();
    }
    std::cout << std::endl;

    this->occupation_matrix = this->determine_occupation();
    this->prev_1s_erg = this->curr_1s_erg;
    this->curr_1s_erg = this->seq_solvers[0].energies(0);
    std::cout << "Total energy: " << this->total_energy() << std::endl;

    this->previous_potential_solver = this->potential_solver;
    this->potential_solver = collocation<numeric_type>(this->zero_function, this->zero_function,
                                                       [&](numeric_type r){return this->colloc_rhs(r);},
                                                       this->knots_pot, {0, N_e});

    this->potential_solver.solve();
    std::cout << "Updated rho solution..." << std::endl;
    this->check_rho();

    if (i == 1) this->second_it_finished = true;
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
            if (occ < 1) break;

            boost::math::quadrature::exp_sinh<numeric_type> integrator;
            auto f = [&](numeric_type r)
                    {return pow(r * this->seq_solvers[l].solution_n(n-l, r), 2) * this->many_body_potential(r);};
            numeric_type V_int = integrator.integrate(f);
            ret += occ * (this->seq_solvers[l].energies(n-l) - 0.5 * V_int); // n - l !
        }
    }
    return ret;
}

template<typename numeric_type>
void atom<numeric_type>::check_rho() {
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

// TODO better knot distribution
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
    // 4 pi epsilon = e = 1 TODO sign?
    return (this->init_finished) ?
        -3.0 * pow(3.0 * abs(this->rho(r)) / (8.0 * numeric_type(M_PI)), 1.0/3.0) : 0.0;
}

template<typename numeric_type> // TODO changed sign
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
    // 4 pi epsilon = e = 1 TODO sign?
    return (this->second_it_finished) ?
        -3.0 * pow(3.0 * abs(this->rho(r)) / (8.0 * numeric_type(M_PI)), 1.0/3.0) : 0.0;
}

template<typename numeric_type> // TODO changed sign
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

#endif //PERIODICTAB_ATOM_H
