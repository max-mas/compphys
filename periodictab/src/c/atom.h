//
// Created by max on 5/7/24.
//
#include <functional>
#include <vector>
#include <cmath>
#include <iostream>

#include <Eigen/Dense>

#include "spherical_seq.h"
#include "collocation.h"

#ifndef PERIODICTAB_ATOM_H
#define PERIODICTAB_ATOM_H

template <typename numeric_type>
class atom {
public:
    unsigned int Z;
    unsigned int max_it_self_consistency;
    unsigned int num_knots;
    unsigned int l_max;
    numeric_type r_max;

    Eigen::MatrixXi occupation_matrix;

    numeric_type electrostatic_potential(numeric_type r);

    numeric_type exchange_potential(numeric_type r);

    numeric_type many_body_potential(numeric_type r);

    numeric_type nuclear_potential(numeric_type r);

    numeric_type mean_field_potential(numeric_type r);

    static numeric_type zero_function(numeric_type r);

    numeric_type colloc_rhs(numeric_type r);

    atom() = default;

    atom(unsigned int chargeNumber, unsigned int maxIt,
         numeric_type rMax, unsigned int numKnots);

private:
    std::vector<spherical_seq<numeric_type>> seq_solvers;
    collocation<numeric_type> potential_solver;

    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> knots;

    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> generate_knots();

    Eigen::MatrixXi determine_occupation();

    bool init_finished = false;
};


template<typename numeric_type>
atom<numeric_type>::atom(unsigned int chargeNumber, unsigned int maxIt,
                         numeric_type rMax, unsigned int numKnots):
Z(chargeNumber), max_it_self_consistency(maxIt), r_max(rMax), num_knots(numKnots) {
    this->l_max = 10; // TODO eh
    this->knots = generate_knots();
    this->seq_solvers.reserve(l_max);

    std::cout << "Initial SEQ solution with V = 0." << std::endl;
    for (int l = 0; l <= this->l_max; l++) {
        std::cout << "l = " << l << std::endl;
        this->seq_solvers[l] = spherical_seq<numeric_type>(this->knots,
                                                   [this](numeric_type r){return this->mean_field_potential(r);}, l);
        this->seq_solvers[l].solve();
    }

    this->occupation_matrix = this->determine_occupation();

    this->potential_solver = collocation<numeric_type>(this->zero_function, this->zero_function,
                                                       [this](numeric_type r){return this->colloc_rhs(r);},
                                                       this->knots, {0, Z}); // TODO BC correct??

    std::cout << "Initial rho solution." << std::endl;
    this->potential_solver.solve();

    this->init_finished = true;
}

// row index l, column index n
template<typename numeric_type>
Eigen::MatrixXi atom<numeric_type>::determine_occupation() {
    Eigen::MatrixXi retmat = Eigen::MatrixXi::Zero(this->l_max + 1, this->knots.size());

    return retmat;
}

// TODO better knot distribution
template<typename numeric_type>
Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> atom<numeric_type>::generate_knots() {
    return Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>::LinSpaced(this->num_knots, 0.0, this->r_max);
}

template<typename numeric_type>
numeric_type atom<numeric_type>::electrostatic_potential(numeric_type r) {
    return (this->init_finished) ? this->potential_solver.solution(r) : 0.0;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::exchange_potential(numeric_type r) {
    // 4 pi epsilon = e = 1
    return -3.0 * pow(3.0 * this->electrostatic_potential(r) / (8.0 * M_PI), 1.0/3.0);
}

template<typename numeric_type>
numeric_type atom<numeric_type>::many_body_potential(numeric_type r) {
    // e = 1
    return 1.0 * this->electrostatic_potential(r) + 1.0 * this->exchange_potential(r);
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
numeric_type atom<numeric_type>::zero_function(numeric_type r) {
    return 0.0;
}

template<typename numeric_type>
numeric_type atom<numeric_type>::colloc_rhs(numeric_type r) {
    numeric_type ret = 0.0;
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots.size(); n++) {
            int occ = this->occupation_matrix(l, n);
            ret += (occ > 0) ? occ * this->seq_solvers[l].solution_n(n, r) : 0.0;
        }
    }
    return -r * ret;
}

#endif //PERIODICTAB_ATOM_H
