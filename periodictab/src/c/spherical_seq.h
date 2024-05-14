/**
 * @file spherical_seq.h
 * @author Max Maschke (m.maschke@tu-bs.de)
 * @brief Templated class that solves the Schrödinger equation for spherically
 * symmetric Problems.
 * @date 2024-04-28
 * 
 * @copyright Copyright (c) 2024 Max Maschke
 * 
 */

#ifndef EIG_SPLINES_SPHERICAL_SEQ_H
#define EIG_SPLINES_SPHERICAL_SEQ_H

// user imports
#include "b_splines.h"
// Eigen imports
#include <Eigen/Dense>
// STL imports
#include <functional>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
//TODO Rm
#include <fenv.h>
#include <boost/math/quadrature/exp_sinh.hpp>

/**
 * @brief Spherical Schrödinger equation solver.
 * Class that solves the 1 particle 3D Schrödinger equation using B-splines. Specifically, the radial part of
 * the wave function R_ln (r) is obtained as an expansion in splines by solving a generalised 
 * Eigenvalue problem.
 * Atomic units are used.
 * @tparam numeric_type Real floating point type.
 */
template <typename numeric_type>
class spherical_seq {
public:
    // order of B-splines k, must be >= 4. polynomial degree = k - 1
    unsigned int spline_order;
    // angular momentum sector
    unsigned int l;

    /**
     * @brief Default constructor
     * Construct a new default spherical seq object.
     */
    spherical_seq() = default;

    /**
     * @brief Constructor.
     * Construct a new default spherical seq object from a specified vector of knot points
     * and a potential V(r). The angular momentum sector l must also be specified.
     * V(r) should return an energy in Hartree.
     * By default, 4th order splines are used. The weights and points for the Gauss-Legendre
     * quadrature can be changed optionally.
     * Generates the matrices H and B for the generalised eigenvalue problem.
     * Appropriate boundary conditions are automatically implemented (P_nl(0) =P _nl(infty) = 0).
     * @param physicalPoints Vector containing the physical points for the splines.
     * @param potential Potential function V(r), [V] != H (Hartree)
     * @param L Angular momentum quantum number, L > 0. Default: 0
     * @param splineOrder Order of the splines to be used. Must be >= 4. Default: 4
     * @param weightsXs Parameters for Gauss-Legendre integration. Uses n=4 weights by default.
     * 
     */
    spherical_seq(const Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> & physicalPoints,
                  const std::function<numeric_type (numeric_type)> & potential,
                  unsigned int L = 0,
                  unsigned int splineOrder = 4,
                  const std::vector<std::pair<numeric_type, numeric_type>> & weightsXs
                  //  = {{1, -1/sqrt(3)}, {1, 1/sqrt(3)}} );

                  //  = {{5.0/9.0, -sqrt(3.0/5.0)}, {5.0/9.0, sqrt(3.0/5.0)}, {8.0/9.0, 0.0}} );

                  = {{(18.0 + sqrt(30.0))/36.0, -sqrt(3.0/7.0 - 2.0/7.0 * sqrt(6.0/5.0))},
                     {(18.0 + sqrt(30.0))/36.0,  sqrt(3.0/7.0 - 2.0/7.0 * sqrt(6.0/5.0))},
                     {(18.0 - sqrt(30.0))/36.0, -sqrt(3.0/7.0 + 2.0/7.0 * sqrt(6.0/5.0))},
                     {(18.0 - sqrt(30.0))/36.0,  sqrt(3.0/7.0 + 2.0/7.0 * sqrt(6.0/5.0))}} );


    /** 
     * Performs the solution of the generalised eigenvalue problem.
     * Must be called before the solutions are accessible.
    */
    void solve();

    /**
     * @brief N-th solution of the SEQn.
     * Returns R_nl (r)
     * @param n n = 0, ..., N-k: Main/radial quantum number.
     * @param r Radial coordinate
     * @return numeric_type 
     */
    numeric_type solution_n(unsigned int n, numeric_type r);

    /**
     * Save the n-th solution to a txt-file. 
     * @param n n = 0, ..., N-k: Main/radial quantum number.
     * @param n_samples Number of r samples for which to save R_nl (r)
     * @param rmin, rmax Range in which to sample the solution.
     * @param path Path to a .txt file
     */
    void save_solution_n(unsigned int n, unsigned int n_samples, numeric_type rmin, numeric_type rmax,
                         const std::string & path);

    /** 
     * Save the generalised eigenvalues to a txt-file.
     * @param path Path to a .txt file
     */
    void save_energies(const std::string & path);

    // spherically symmetric potential V(r)
    std::function<numeric_type (numeric_type)> V;

    // physical points vector, first element must be 0
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> physical_points;

    // generalised eigenvectors
    Eigen::Matrix<numeric_type, Eigen::Dynamic, Eigen::Dynamic> solution_coeffs;
    // generalised eigenvalues
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> energies;

private:
    // flag to check if solved
    bool solved = false;

    // integrated Hamiltonian matrix
    Eigen::Matrix<numeric_type, Eigen::Dynamic, Eigen::Dynamic> H;
    // integrated RHS matrix
    Eigen::Matrix<numeric_type, Eigen::Dynamic, Eigen::Dynamic> B;
    // splines object
    b_splines<numeric_type> splines;

    // Gauss weights and points
    std::vector<std::pair<numeric_type, numeric_type>> weights_xs;
    
    /** 
     * Performs Gaussian integration of a function f on an interval [a, b] using the weights stored
     * as class members. For n weights, this is exact for polynomials up to degree n-1.
     * @param a, b Interval bounds
     * @param f Single variable function.
     * @return numeric_type Integral \int_a^b f(x) dx
     */
    numeric_type gauss_int(numeric_type a, numeric_type b,
                           const std::function<numeric_type (numeric_type)> & f);
    
    /**
     * Action of the Hamiltonian on the i-th spline.
     * @param i Spline index.
     * @param r Radial coordinate.
     * @return numeric_type 
     */
    numeric_type H_B_i(unsigned int i, numeric_type r);
    
    /**
     * Integrand of the matrix element h_ij.
     * @param i, j Matrix element indices, 0 <= i,j <= N-k
     * @param r Radial coordinate
     * @return numeric_type 
     */
    numeric_type h_ij(unsigned int i, unsigned int j, numeric_type r);
    
    /**
     * Integrand of the matrix element b_ij.
     * @param i, j Matrix element indices, 0 <= i,j <= N-k
     * @param r Radial coordinate
     * @return numeric_type 
     */
    numeric_type b_ij(unsigned int i, unsigned int j, numeric_type r);
    
    /**
     * Returns minimum of two integers i and j.
     * @param i 
     * @param j 
     * @return int 
     */
    int imin(int i, int j);

    /**
     * Returns maximum of two integers i and j.
     * @param i 
     * @param j 
     * @return int 
     */
    int imax(int i, int j);

    /**
     * Normalises the states according to \int_0^\infty P_nl^2(r) dr = 1
     */
    void normalise_states();
};

template<typename numeric_type>
spherical_seq<numeric_type>::spherical_seq(const Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> &physicalPoints,
                                           const std::function<numeric_type(numeric_type)> &potential,
                                           unsigned int L, unsigned int splineOrder,
                                           const std::vector<std::pair<numeric_type, numeric_type>> & weightsXs):
physical_points(physicalPoints), V(potential), spline_order(splineOrder), l(L), weights_xs(weightsXs) {
    if (physicalPoints(0) != 0.0) {
        throw std::runtime_error("For valid results, the knot points must start at 0.");
    }

    this->splines = b_splines<numeric_type>(this->spline_order, this->physical_points);

    unsigned int N = this->splines.num_knots;
    unsigned int k = this->spline_order;

    this->H = Eigen::Matrix<numeric_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(N - k - 2, N - k - 2);
    this->B = Eigen::Matrix<numeric_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(N - k - 2, N - k - 2);
    // we don't use the first and last spline to enforce bcs
    for (int i = 1; i < N - k - 1; i++) {
        for (int j = 1; j < N - k - 1; j++) {
            numeric_type h = 0;
            numeric_type b = 0;
            int m_min = imax(i, j);
            int m_max = imin(i, j) + k - 1;
            for (int m = m_min; m <= m_max; m++) {
                numeric_type t_m = this->splines.knot_points(m);
                numeric_type t_m1 = this->splines.knot_points(m+1);
                h += gauss_int(t_m, t_m1, [&i, &j, this](numeric_type r){return this->h_ij(i, j, r);});
                b += gauss_int(t_m, t_m1, [&i, &j, this](numeric_type r){return this->b_ij(i, j, r);});
            }
            H(i-1, j-1) = h;
            B(i-1, j-1) = b;
        }
    }
}


template<typename numeric_type>
numeric_type spherical_seq<numeric_type>::solution_n(unsigned int n, numeric_type r) {
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> coeffs = this->solution_coeffs.col(n);
    if (r == 0) {
        r = std::numeric_limits<numeric_type>::epsilon(); //todo changed from min()
    }
    numeric_type ret = 0;
    for (int i = 1; i < this->splines.num_knots - this->spline_order - 2; i++) {
        ret += coeffs(i-1) * this->splines.B_i(i, r); // *gravestone emoji*
    }
    // dont forget to divide by r, prevent div by 0
    return ret/r;
}

template<typename numeric_type>
void spherical_seq<numeric_type>::save_solution_n(unsigned int n, unsigned int n_samples, numeric_type rmin,
                                                  numeric_type rmax, const std::string & path) {
    Eigen::VectorXd points = Eigen::VectorXd::LinSpaced(n_samples, rmin, rmax);

    std::ofstream file;
    file.open(path);
    // write
    for (int j = 0; j < n_samples; j++) {
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
             << std::scientific << points[j] << ","
             << std::scientific << solution_n(n, points[j]) << std::endl;
    }
    // don't forget to clean up :)
    file.close();
}

template<typename numeric_type>
void spherical_seq<numeric_type>::save_energies(const std::string &path) {
    std::ofstream file;
    file.open(path);
    // write
    for (int j = 0; j < this->energies.size(); j++) {
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
             << std::scientific << j+1+this->l << ","
             << std::scientific << energies(j) << std::endl;
    }
    // don't forget to clean up :)
    file.close();
}

template<typename numeric_type>
void spherical_seq<numeric_type>::solve() {
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<numeric_type, Eigen::Dynamic, Eigen::Dynamic>>
        es(this->H, this->B, Eigen::ComputeEigenvectors);
    this->energies = es.eigenvalues();
    this->solution_coeffs = es.eigenvectors();
    this->normalise_states();
    this->solved = true;
}

template<typename numeric_type>
void spherical_seq<numeric_type>::normalise_states() {
    for (int i = 0; i < this->solution_coeffs.cols(); i++) {
        if (this->energies(i) > 0) break; // ordered! only normalise bound states, rest: who cares?

        numeric_type integral2 = this->solution_coeffs.col(i).transpose() * this->B * this->solution_coeffs.col(i);

        this->solution_coeffs.col(i) /= sqrt(integral2);
    }
}

template<typename numeric_type>
numeric_type spherical_seq<numeric_type>::gauss_int(numeric_type a, numeric_type b,
                                                        const std::function<numeric_type(numeric_type)> &f) {
    numeric_type ret = 0.0;
    for (std::pair<numeric_type, numeric_type> w_x: this->weights_xs) {
        ret += w_x.first * f((b-a)/2 * w_x.second + (a+b)/2);
    }
    ret *= (b-a) / 2;
    return ret;

}

template<typename numeric_type> //TODO check change valid
numeric_type spherical_seq<numeric_type>::H_B_i(unsigned int i, numeric_type r) {
    if (r <= 0.0) r = std::numeric_limits<numeric_type>::epsilon();
    numeric_type ret = 0.0;
    ret += - this->splines.B_i_xx(i, r) / 2;
    ret += (this->l * (this->l + 1) / (2 * (pow(r, 2))) + this->V(r)) * this->splines.B_i(i, r);

    return ret;
}

template<typename numeric_type>
numeric_type spherical_seq<numeric_type>::h_ij(unsigned int i, unsigned int j, numeric_type r) {
    return this->splines.B_i(i, r) * this->H_B_i(j, r);
}

template<typename numeric_type>
numeric_type spherical_seq<numeric_type>::b_ij(unsigned int i, unsigned int j, numeric_type r) {
    return this->splines.B_i(i, r) * this->splines.B_i(j, r);
}

template<typename numeric_type>
int spherical_seq<numeric_type>::imin(int i, int j) {
    return (i < j) ? i : j;
}

template<typename numeric_type>
int spherical_seq<numeric_type>::imax(int i, int j) {
    return (i > j) ? i : j;
}



#endif //EIG_SPLINES_SPHERICAL_SEQ_H
