/**
 * @file b_splines.h
 * @author Max Maschke (m.maschke@tu-bs.de)
 * @brief 
 * @version 0.1
 * @date 2024-04-28
 * 
 * @copyright Copyright (c) 2024 Max Maschke
 * 
 */

#ifndef POISSON_B_SPLINES_H
#define POISSON_B_SPLINES_H

// Eigen import
#include <Eigen/Dense>

// STL imports
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <functional>
#include <stdexcept>

/**
 * Class that implements b-splines of order k, accessed by calling B_i(i, x).
 * Provides a convenience function that saves B_i(x) to a file for a given i and discrete interval.
 * By default, adds an appropriate number of ghost points at the boundaries, which is needed to ensure the
 * B_i are a local basis on the interval. If this is overridden, the user should know what they're doing :).
 * the user must ensure to use a correct number
 */

class b_splines {
public:
    int order_k{}; // note: this is equal to n + 1 where n is the polynomial degree of the B_i_k.
    int num_ghosts{}; // Number of ghost points, not set by the user.
    int num_knots{}; // includes ghosts
    Eigen::VectorXd knot_points; // if double = double, this is eq. to VectorXd

    /**
     * Constructor. By default, appends correct number of ghost points.
     * @param orderK spline order = n + 1 where n is the polynomial degree
     * @param knotPoints (not necessarily strictly) increasing vector of knot points
     */
    b_splines(int orderK, const Eigen::VectorXd & knotPoints);

    b_splines() = default;

    /**
     * Public function that returns the ith spline at x.
     * @param i Number of the spline, valid from 0 to num_knots - order_k.
     * @param x
     * @return B_i(x)
     */
    double B_i(int i, double x);

    /**
     * Returns the derivative of the ith spline at x.
     * @param i Number of the spline, valid from 0 to num_knots - order_k.
     * @param x
     * @return B_i(x)
     */
    double B_i_x(int i, double x);

    /**
     * Returns the second derivative of the ith spline at x.
     * @param i Number of the spline, valid from 0 to num_knots - order_k.
     * @param x
     * @return B_i(x)
     */
    double B_i_xx(int i, double x);

    /**
     * Convenience function that creates the directory at path and saves the ith spline on the interval
     * [-xmin, xmax] with n_samples evaluation points.
     * @param i Number of the spline, valid from 0 to num_knots - order_k.
     * @param n_samples
     * @param xmin
     * @param xmax
     * @param path Path to a directory to save to (not a file!)
     * @param deriv Default 0: bare spline, 1: 1st derivative, 2: 2nd derivative, else: bare
     */
    void save_B_i(int i, int n_samples, double xmin, double xmax, const std::string & path, int deriv = 0);

private:
    /**
     * Internal function that calculates the splines by recursion in k.
     * @param i Number of the spline, valid from 0 to num_knots - order_k.
     * @param k Spline order k, valid from 1 to order_k
     * @param x
     * @return B_i_k(x)
     */
    double B_i_k(int i, int k, double x);
};

#endif //POISSON_B_SPLINES_H
