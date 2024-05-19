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
 */
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
    spherical_seq(const Eigen::VectorXd & physicalPoints,
                  const std::function<double (double)> & potential,
                  unsigned int L = 0,
                  unsigned int splineOrder = 4,
                  const std::vector<std::pair<double, double>> & weightsXs
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
     * @return double 
     */
    double solution_n(unsigned int n, double r);

    /**
     * Save the n-th solution to a txt-file. 
     * @param n n = 0, ..., N-k: Main/radial quantum number.
     * @param n_samples Number of r samples for which to save R_nl (r)
     * @param rmin, rmax Range in which to sample the solution.
     * @param path Path to a .txt file
     */
    void save_solution_n(unsigned int n, unsigned int n_samples, double rmin, double rmax,
                         const std::string & path);

    /** 
     * Save the generalised eigenvalues to a txt-file.
     * @param path Path to a .txt file
     */
    void save_energies(const std::string & path);

    // spherically symmetric potential V(r)
    std::function<double (double)> V;

    // physical points vector, first element must be 0
    Eigen::VectorXd physical_points;

    // generalised eigenvectors
    Eigen::MatrixXd solution_coeffs;
    // generalised eigenvalues
    Eigen::VectorXd energies;

private:
    // flag to check if solved
    bool solved = false;

    // integrated Hamiltonian matrix
    Eigen::MatrixXd H;
    // integrated RHS matrix
    Eigen::MatrixXd B;
    // splines object
    b_splines splines;

    // Gauss weights and points
    std::vector<std::pair<double, double>> weights_xs;
    
    /** 
     * Performs Gaussian integration of a function f on an interval [a, b] using the weights stored
     * as class members. For n weights, this is exact for polynomials up to degree n-1.
     * @param a, b Interval bounds
     * @param f Single variable function.
     * @return double Integral \int_a^b f(x) dx
     */
    double gauss_int(double a, double b,
                           const std::function<double (double)> & f);
    
    /**
     * Action of the Hamiltonian on the i-th spline.
     * @param i Spline index.
     * @param r Radial coordinate.
     * @return double 
     */
    double H_B_i(unsigned int i, double r);
    
    /**
     * Integrand of the matrix element h_ij.
     * @param i, j Matrix element indices, 0 <= i,j <= N-k
     * @param r Radial coordinate
     * @return double 
     */
    double h_ij(unsigned int i, unsigned int j, double r);
    
    /**
     * Integrand of the matrix element b_ij.
     * @param i, j Matrix element indices, 0 <= i,j <= N-k
     * @param r Radial coordinate
     * @return double 
     */
    double b_ij(unsigned int i, unsigned int j, double r);
    
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

#endif //EIG_SPLINES_SPHERICAL_SEQ_H
