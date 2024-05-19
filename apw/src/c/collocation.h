/**
 * @file collocation.h
 * @author Max Maschke (m.maschke@tu-bs.de)
 * @brief Templated class that solves 2nd order one dimensional boundary value problems,
 * such as Poisson's equation for spherically symmetrical charge distributions.
 * @date 2024-04-28
 *
 * @copyright Copyright (c) 2024 Max Maschke
 *
 */

#ifndef POISSON_COLLOCATION_H
#define POISSON_COLLOCATION_H

#include "b_splines.h"

#include <Eigen/Dense>

#include <functional>
#include <tuple>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

/**
 * Class that implements the collocation method for a second order ODE on a real interval with
 * boundary conditions at the ends.
 * Suitable for equations of the form:
 * f'' + p * f' + q * f = g
 * E.g. Poisson's eqn:
 * f'' = g
 * Provides constructors that generate the discretisation on the interval, including extra points near
 * discontinuities in the RHS if applicable.
 */
class collocation {
public:
    // prefactor of the 1st derivative
    std::function<double (double)> p;
    // prefactor of the linear term
    std::function<double (double)> q;
    // RHS
    std::function<double (double)> g;
    // solution interval
    std::pair<double, double> interval;
    // boundary conditions, here: f on the boundaries of the interval
    std::pair<double, double> boundary_conditions;
    // number of physical discretisation points
    int num_physical_points;

    /**
     * Constructor for the collocation class if the discretisation on the interval is supplied externally.
     * @param P derivative prefactor
     * @param Q linear prefactor
     * @param G RHS
     * @param physical_knots physical points in the discretisation (ghost points are internally appended)
     * @param Boundary_Conditions boundary conditions
     */
    collocation(const std::function<double (double)> & P,
                const std::function<double (double)> & Q,
                const std::function<double (double)> & G,
                const Eigen::VectorXd & physical_knots,
                const std::pair<double, double> & Boundary_Conditions);

    /**
     * Constructor for the collocation class if the discretisation on the interval is
     * to be generated with linear spacing internally.
     * @param P derivative prefactor
     * @param Q linear prefactor
     * @param G RHS
     * @param Interval interval
     * @param Boundary_Conditions boundary conditions
     * @param Num_Physical_Points number of physical points to be included in the discretisation
     */
    collocation(const std::function<double (double)> & P,
                const std::function<double (double)> & Q,
                const std::function<double (double)> & G,
                const std::pair<double, double> & Interval,
                const std::pair<double, double> & Boundary_Conditions,
                int Num_Physical_Points);

    /**
     * Constructor for the collocation class if the discretisation on the interval is
     * to be generated with linear spacing internally and there are points of discontinuity in the RHS.
     * @param P derivative prefactor
     * @param Q linear prefactor
     * @param G RHS
     * @param Interval interval
     * @param Boundary_Conditions boundary conditions
     * @param Critical_Points points of discontinuity in the RHS where the numerics benefit from a tighter
     *                        grouping of points
     * @param Num_Physical_Points number of physical points to be included in the discretisation
     */
    collocation(const std::function<double (double)> & P,
                const std::function<double (double)> & Q,
                const std::function<double (double)> & G,
                const std::pair<double, double> & Interval,
                const std::pair<double, double> & Boundary_Conditions,
                const Eigen::VectorXd & Critical_Points,
                int Num_Physical_Points);

    /**
     * Default constructor.
     */
    collocation() = default;

    /**
     * Must be called before results are accessible.
     * Gets the solution basis coefficients using LU factorisation.
     */
    void solve();

    /**
     * Returns the previously calculated solution at x.
     * @param x
     * @return
     */
    double solution(double x);

    /**
     * Saves the previously calculated solution on a discrete sample interval.
     * @param xmin
     * @param xmax
     * @param num_xs
     * @param path Path to a txt file or equivalent
     */
    void save_solution(double xmin, double xmax, int num_xs, const std::string & path);

private:
    // if false, solution() and save_solution() are inaccessible / throw exceptions
    bool solved = false;

    // b spline object of order 4
    b_splines splines;

    // coefficient matrix, is destroyed in place by solve()!
    Eigen::MatrixXd coeff_mat;
    Eigen::VectorXd rhs_vector;
    Eigen::VectorXd solution_coeffs;

};


#endif //POISSON_COLLOCATION_H
