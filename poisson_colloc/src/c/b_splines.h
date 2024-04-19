//
// Created by max on 4/19/24.
//

#ifndef POISSON_B_SPLINES_H
#define POISSON_B_SPLINES_H

#include <vector>
#include <Eigen/Dense>
#include <stdexcept>

template <typename numeric_type>
class b_splines {
public:
    int order_k;
    int num_ghosts; //TODO why not const?
    int num_knots; // includes ghosts
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> knot_points; // if numeric_type = double, this is eq. to VectorXd TODO can I make this const?

    b_splines(int orderK, const Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> & knotPoints,
              bool appendGhosts = true);

    numeric_type B_i_k(int i, int k, numeric_type x);
};

template<typename numeric_type>
b_splines<numeric_type>::b_splines(int orderK, const Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> &knotPoints,
                                   bool appendGhosts): order_k(orderK) {
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

    } else {
        num_knots = knotPoints.size();
        num_ghosts = 2 * (orderK - 1);
        knot_points = Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>(knotPoints);
    }
}

template <typename numeric_type>
numeric_type b_splines<numeric_type>::B_i_k(int i, int k, numeric_type x) {
    if (k == 1) {
        return (knot_points(i) <= x < knot_points(i+1)) ? numeric_type(1.0) : numeric_type(0.0);
    } else {
        return (x - knot_points(i)) / (knot_points(i + order_k - 1) - knot_points(i) + numeric_type(1e-15)) * B_i_k(i, k-1, x)
            + (knot_points(i + order_k) - x) / (knot_points(i + order_k) - knot_points(i + 1) + numeric_type(1e-15)) * B_i_k(i+1, k-1, x);
    }
}


#endif //POISSON_B_SPLINES_H
