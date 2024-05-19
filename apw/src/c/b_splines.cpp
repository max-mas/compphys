#include "b_splines.h"

b_splines::b_splines(int orderK, const Eigen::VectorXd &knotPoints):
order_k(orderK) {
    if (orderK < 1) {
        throw std::runtime_error("B splines cannot have order < 1.");
    }
    // normal behaviour: append ghost points automatically

    num_ghosts = 2 * (orderK - 1);
    double t_0 = knotPoints(0);
    double t_f = knotPoints(Eigen::indexing::last);
    num_knots = knotPoints.size() + num_ghosts;

    knot_points = Eigen::VectorXd(num_knots);
    // initialise knot points with appended ghosts
    for (int i = 0; i < num_knots; i++) {
        if (i <= order_k - 1) { knot_points(i) = t_0; }
        else if (i >= num_knots - (order_k - 1)) { knot_points(i) = t_f; }
        else {
            knot_points(i) = knotPoints(i - (order_k - 1));
            if (knot_points(i) < knot_points(i - 1)) {
                throw std::runtime_error("Knot points must be ascending!");
            }
        }
    }
}


double b_splines::B_i(int i, double x) {
    // make sure B_i_k is called with the correct k.
    return B_i_k(i, order_k, x);
}


double b_splines::B_i_x(int i, double x) {
    if (order_k < 3) {
        throw std::runtime_error("B splines are only differentiable for k >= 3.");
    }
    double k1 = (knot_points(i + order_k - 1) - knot_points(i));
    double k2 = (knot_points(i + order_k) - knot_points(i+1));
    return (order_k - 1) * ((k1 != 0) ? (B_i_k(i  , order_k - 1, x) / k1) : double(0.0))
          -(order_k - 1) * ((k2 != 0) ? (B_i_k(i+1, order_k - 1, x) / k2) : double(0.0));
}


double b_splines::B_i_xx(int i, double x) {
    if (order_k < 3) {
        throw std::runtime_error("B splines are only differentiable twice for k >= 4.");
    }
    const int & k = order_k;
    double k1 = ((knot_points(i + k - 1) - knot_points(i)) * (knot_points(i + k - 2) - knot_points(i)));
    double k2 = ((knot_points(i + k - 1) - knot_points(i)) * (knot_points(i + k - 1) - knot_points(i+1)));
    double k3 = ((knot_points(i + k) - knot_points(i+1)) * (knot_points(i + k - 1) - knot_points(i+1)));
    double k4 = ((knot_points(i + k) - knot_points(i+1)) * (knot_points(i + k) - knot_points(i+2)));
    return ((k1 != 0) ? ((k - 1) * (k - 2) * B_i_k(i, k-2, x) / k1) : double(0.0))
          -((k2 != 0) ? ((k - 1) * (k - 2) * B_i_k(i+1, k-2, x) / k2) : double(0.0))
          -((k3 != 0) ? ((k - 1) * (k - 2) * B_i_k(i+1, k-2, x) / k3) : double(0.0))
          +((k4 != 0) ? ((k - 1) * (k - 2) * B_i_k(i+2, k-2, x) / k4) : double(0.0));
}


double b_splines::B_i_k(int i, int k, double x) {
    if (k < 1) {
        throw std::runtime_error("B splines cannot have order < 1."); // this should never happen
    }
    if (k == 1 and i == num_knots-1) {
        // last spline is special, use cox de boor convention
        return (knot_points(i) <= x and x <= knot_points(i+1)) ? double(1.0) : double(0.0);
    }
    if (k == 1) {
        // check if x in interval [t_i, t_i+1)
        return (knot_points(i) <= x and x < knot_points(i+1)) ? double(1.0) : double(0.0);
    } else {
        double k1 = (knot_points(i + k - 1) - knot_points(i));
        double k2 = (knot_points(i + k) - knot_points(i + 1));
        return ((k1 != 0) and (i + k - 1 < num_knots) ? (x - knot_points(i)) / k1 * B_i_k(i, k-1, x) : double(0.0))
             + ((k2 != 0) and (i + k < num_knots) ? (knot_points(i + k) - x) / k2 * B_i_k(i+1, k-1, x) : double(0.0));
    }
}

// path must be a dir not a file

void b_splines::save_B_i(int i, int n_samples, double xmin, double xmax,
                                       const std::string &path,
                                       int deriv) {
    // linspaced xs
    Eigen::VectorXd xs =
            Eigen::VectorXd::LinSpaced(n_samples, xmin, xmax);
    // store Bs
    Eigen::VectorXd Bs(n_samples);

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