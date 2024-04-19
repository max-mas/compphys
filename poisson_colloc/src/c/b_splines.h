//
// Created by max on 4/19/24.
//

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

/**
 * Class that implements b-splines of order k, accessed by calling B_i(i, x).
 * Provides a convenience function that saves B_i(x) to a file for a given i and discrete interval.
 * By default, adds an appropriate number of ghost points at the boundaries, which is needed to ensure the
 * B_i are a local basis on the interval. If this is overridden, the user should know what they're doing :).
 * the user must ensure to use a correct number
 * @tparam numeric_type Numeric type like float, double that has an
 *                      order relation "<" (this generally rules out complex numbers!)
 */
template <typename numeric_type>
class b_splines {
public:
    int order_k; // note: this is equal to n + 1 where n is the polynomial degree of the B_i_k.
    int num_ghosts; // Number of ghost points, not set by the user.
    int num_knots; // includes ghosts
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> knot_points; // if numeric_type = double, this is eq. to VectorXd

    /**
     * Constructor. By default, appends correct number of ghost points.
     * @param orderK spline order = n + 1 where n is the polynomial degree
     * @param knotPoints (not necessarily strictly) increasing vector of knot points
     * @param appendGhosts true by default
     */
    b_splines(int orderK, const Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> & knotPoints,
              bool appendGhosts = true);

    /**
     * Public function that returns the ith spline at x.
     * @param i Number of the spline, valid from 0 to num_knots - order_k.
     * @param x
     * @return B_i(x)
     */
    numeric_type B_i(int i, numeric_type x);

    /**
     * Returns the derivative of the ith spline at x.
     * @param i Number of the spline, valid from 0 to num_knots - order_k.
     * @param x
     * @return B_i(x)
     */
    numeric_type B_i_x(int i, numeric_type x);

    /**
     * Returns the second derivative of the ith spline at x.
     * @param i Number of the spline, valid from 0 to num_knots - order_k.
     * @param x
     * @return B_i(x)
     */
    numeric_type B_i_xx(int i, numeric_type x);

    /**
     * Convenience function that creates the directory at path and saves the ith spline on the interval
     * [-xmin, xmax] with n_samples evaluation points.
     * @param i Number of the spline, valid from 0 to num_knots - order_k.
     * @param n_samples
     * @param xmin
     * @param xmax
     * @param path Path to a directory to save to (not a file!)
     */
    void save_B_i(int i, int n_samples, numeric_type xmin, numeric_type xmax, const std::string & path, int deriv = 0);

private:
    /**
     * Internal function that calculates the splines by recursion in k.
     * @param i Number of the spline, valid from 0 to num_knots - order_k.
     * @param k Spline order k, valid from 1 to order_k
     * @param x
     * @return B_i_k(x)
     */
    numeric_type B_i_k(int i, int k, numeric_type x);
};


template<typename numeric_type>
b_splines<numeric_type>::b_splines(int orderK, const Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> &knotPoints,
                                   bool appendGhosts): order_k(orderK) {
    // normal behaviour: append ghost points automatically
    if (appendGhosts) {
        num_ghosts = 2 * (orderK - 1);
        numeric_type t_0 = knotPoints(0);
        numeric_type t_f = knotPoints(Eigen::indexing::last);
        num_knots = knotPoints.size() + num_ghosts;

        knot_points = Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>(num_knots);
        // initialise knot points with appended ghosts
        for (int i = 0; i < num_knots; i++) {
            if (i <= order_k - 1) {knot_points(i) = t_0;}
            else if (i >= num_knots - (order_k - 1)) {knot_points(i) = t_f;}
            else {knot_points(i) = knotPoints(i - (order_k - 1));}
        }

    // creation with ghost points already included
    } else {
        num_knots = knotPoints.size();
        num_ghosts = 2 * (orderK - 1);
        knot_points = Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>(knotPoints);
    }
}

template <typename numeric_type>
numeric_type b_splines<numeric_type>::B_i(int i, numeric_type x) {
    // make sure B_i_k is called with the correct k.
    return B_i_k(i, order_k, x);
}

template<typename numeric_type>
numeric_type b_splines<numeric_type>::B_i_x(int i, numeric_type x) {
    numeric_type k1 = (knot_points(i + order_k - 1) - knot_points(i));
    numeric_type k2 = (knot_points(i + order_k) - knot_points(i+1));
    return (order_k - 1) * ((k1 != 0) ? (B_i_k(i  , order_k - 1, x) / k1) : numeric_type(0.0))
          -(order_k - 1) * ((k2 != 0) ? (B_i_k(i+1, order_k - 1, x) / k2) : numeric_type(0.0));
}

template<typename numeric_type>
numeric_type b_splines<numeric_type>::B_i_xx(int i, numeric_type x) {
    const int & k = order_k;
    numeric_type k1 = ((knot_points(i + k - 1) - knot_points(i)) * (knot_points(i + k - 2) - knot_points(i)));
    numeric_type k2 = ((knot_points(i + k - 1) - knot_points(i)) * (knot_points(i + k - 1) - knot_points(i+1)));
    numeric_type k3 = ((knot_points(i + k) - knot_points(i+1)) * (knot_points(i + k - 1) - knot_points(i+1)));
    numeric_type k4 = ((knot_points(i + k) - knot_points(i+1)) * (knot_points(i + k) - knot_points(i+2)));
    return ((k1 != 0) ? ((k - 1) * (k - 2) * B_i_k(i, k-2, x) / k1) : numeric_type(0.0))
          -((k2 != 0) ? ((k - 1) * (k - 2) * B_i_k(i+1, k-2, x) / k2) : numeric_type(0.0))
          -((k3 != 0) ? ((k - 1) * (k - 2) * B_i_k(i+1, k-2, x) / k3) : numeric_type(0.0))
          +((k4 != 0) ? ((k - 1) * (k - 2) * B_i_k(i+2, k-2, x) / k4) : numeric_type(0.0));
}

template <typename numeric_type>
numeric_type b_splines<numeric_type>::B_i_k(int i, int k, numeric_type x) {
    if (k == 1 and i == num_knots-1) {
        // last spline is special, use cox de boor convention
        return (knot_points(i) <= x and x <= knot_points(i+1)) ? numeric_type(1.0) : numeric_type(0.0);
    }
    if (k == 1) {
        // check if x in interval [t_i, t_i+1)
        return (knot_points(i) <= x and x < knot_points(i+1)) ? numeric_type(1.0) : numeric_type(0.0);
    } else {
        numeric_type k1 = (knot_points(i + k - 1) - knot_points(i));
        numeric_type k2 = (knot_points(i + k) - knot_points(i + 1));
        return ((k1 != 0) and (i + k - 1 < num_knots) ? (x - knot_points(i)) / k1 * B_i_k(i, k-1, x) : numeric_type(0.0))
             + ((k2 != 0) and (i + k < num_knots) ? (knot_points(i + k) - x) / k2 * B_i_k(i+1, k-1, x) : numeric_type(0.0));
    }
}

// path must be a dir not a file
template<typename numeric_type>
void b_splines<numeric_type>::save_B_i(int i, int n_samples, numeric_type xmin, numeric_type xmax,
                                       const std::string &path,
                                       int deriv) {
    // linspaced xs
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> xs =
            Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>::LinSpaced(n_samples, xmin, xmax);
    // store Bs
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> Bs(n_samples);

    for (int j = 0; j < n_samples; j++) {
        if (deriv == 1) { Bs(j) = B_i_x(i, xs(j)); }
        else if (deriv == 2) { Bs(j) = B_i_xx(i, xs(j)); }
        else { Bs(j) = B_i(i, xs(j)); }
    }
    // mkdir
    std::filesystem::create_directory(path);
    std::ofstream file;
    file.open(path + "B_" + std::to_string(i) + ".txt");
    // write
    for (int j = 0; j < n_samples; j++) {
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
             << std::scientific << xs[j] << ","
             << std::scientific << Bs[j] << std::endl;
    }
    // don't forget to clean up :)
    file.close();
}

#endif //POISSON_B_SPLINES_H
