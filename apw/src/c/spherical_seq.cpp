#include "spherical_seq.h"

spherical_seq::spherical_seq(const Eigen::VectorXd &physicalPoints,
                                           const std::function<double(double)> &potential,
                                           unsigned int L, unsigned int splineOrder,
                                           const std::vector<std::pair<double, double>> & weightsXs):
physical_points(physicalPoints), V(potential), spline_order(splineOrder), l(L), weights_xs(weightsXs) {
    if (physicalPoints(0) != 0.0) {
        throw std::runtime_error("For valid results, the knot points must start at 0.");
    }

    this->splines = b_splines(this->spline_order, this->physical_points);

    unsigned int N = this->splines.num_knots;
    unsigned int k = this->spline_order;

    this->H = Eigen::MatrixXd::Zero(N - k - 2, N - k - 2);
    this->B = Eigen::MatrixXd::Zero(N - k - 2, N - k - 2);
    // we don't use the first and last spline to enforce bcs
    for (int i = 1; i < N - k - 1; i++) {
        for (int j = 1; j < N - k - 1; j++) {
            double h = 0;
            double b = 0;
            int m_min = imax(i, j);
            int m_max = imin(i, j) + k - 1;
            for (int m = m_min; m <= m_max; m++) {
                double t_m = this->splines.knot_points(m);
                double t_m1 = this->splines.knot_points(m+1);
                h += gauss_int(t_m, t_m1, [&i, &j, this](double r){return this->h_ij(i, j, r);});
                b += gauss_int(t_m, t_m1, [&i, &j, this](double r){return this->b_ij(i, j, r);});
            }
            H(i-1, j-1) = h;
            B(i-1, j-1) = b;
        }
    }
}



double spherical_seq::solution_n(unsigned int n, double r) {
    Eigen::VectorXd coeffs = this->solution_coeffs.col(n);
    if (r == 0) {
        r = std::numeric_limits<double>::epsilon(); //todo changed from min()
    }
    double ret = 0;
    for (int i = 1; i < this->splines.num_knots - this->spline_order - 2; i++) {
        ret += coeffs(i-1) * this->splines.B_i(i, r); // *gravestone emoji*
    }
    // dont forget to divide by r, prevent div by 0
    return ret/r;
}


void spherical_seq::save_solution_n(unsigned int n, unsigned int n_samples, double rmin,
                                                  double rmax, const std::string & path) {
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


void spherical_seq::save_energies(const std::string &path) {
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


void spherical_seq::solve() {
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd>
        es(this->H, this->B, Eigen::ComputeEigenvectors);
    this->energies = es.eigenvalues();
    this->solution_coeffs = es.eigenvectors();
    this->normalise_states();
    this->solved = true;
}


void spherical_seq::normalise_states() {
    for (int i = 0; i < this->solution_coeffs.cols(); i++) {
        if (this->energies(i) > 0) break; // ordered! only normalise bound states, rest: who cares?

        double integral2 = this->solution_coeffs.col(i).transpose() * this->B * this->solution_coeffs.col(i);

        this->solution_coeffs.col(i) /= sqrt(integral2);
    }
}


double spherical_seq::gauss_int(double a, double b,
                                                        const std::function<double(double)> &f) {
    double ret = 0.0;
    for (std::pair<double, double> w_x: this->weights_xs) {
        ret += w_x.first * f((b-a)/2 * w_x.second + (a+b)/2);
    }
    ret *= (b-a) / 2;
    return ret;

}

 //TODO check change valid
double spherical_seq::H_B_i(unsigned int i, double r) {
    if (r <= 0.0) r = std::numeric_limits<double>::epsilon();
    double ret = 0.0;
    ret += - this->splines.B_i_xx(i, r) / 2;
    ret += (this->l * (this->l + 1) / (2 * (pow(r, 2))) + this->V(r)) * this->splines.B_i(i, r);

    return ret;
}


double spherical_seq::h_ij(unsigned int i, unsigned int j, double r) {
    return this->splines.B_i(i, r) * this->H_B_i(j, r);
}


double spherical_seq::b_ij(unsigned int i, unsigned int j, double r) {
    return this->splines.B_i(i, r) * this->splines.B_i(j, r);
}


int spherical_seq::imin(int i, int j) {
    return (i < j) ? i : j;
}


int spherical_seq::imax(int i, int j) {
    return (i > j) ? i : j;
}