//
// Created by max on 4/24/24.
//

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

template <typename numeric_type>
class spherical_seq {
public:
    // order of B-splines k, must be >= 4. polynomial degree = k - 1
    unsigned int spline_order;
    // angular momentum sector
    unsigned int l;

    spherical_seq() = default;

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


    void solve();

    numeric_type solution_n(unsigned int n, numeric_type r);

    void save_solution_n(unsigned int n, unsigned int n_samples, numeric_type rmin, numeric_type rmax,
                         const std::string & path);

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
    // Gaussian integration
    numeric_type gauss_int(numeric_type a, numeric_type b,
                           const std::function<numeric_type (numeric_type)> & f);
    // Hamiltonian on spline
    numeric_type H_B_i(unsigned int i, numeric_type r);
    // integrand lhs
    numeric_type h_ij(unsigned int i, unsigned int j, numeric_type r);
    // integrand rhs
    numeric_type b_ij(unsigned int i, unsigned int j, numeric_type r);
    //
    int imin(int i, int j);
    int imax(int i, int j);

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

    //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT); // TODO RM
    // we don't use the first and last spline to enforce bcs
    for (int i = 1; i < N - k - 1; i++) {
        for (int j = 1; j < N - k - 1; j++) {
            numeric_type h = 0;
            numeric_type b = 0;
            int m_min = imax(i, j);
            int m_max = imin(i, j) + k - 1;
            for (int m = m_min; m <= m_max; m++) {
            //for (int m =0 ; m < N-1; m++) {
                //std::cout << m+1 << ";" << this->physical_points.size() << std::endl;
                //if (m == 0 or m+1 >= this->physical_points.size()-1) {continue;}
                numeric_type t_m = this->splines.knot_points(m);
                numeric_type t_m1 = this->splines.knot_points(m+1);
                h += gauss_int(t_m, t_m1, [&i, &j, this](numeric_type r){return this->h_ij(i, j, r);});
                b += gauss_int(t_m, t_m1, [&i, &j, this](numeric_type r){return this->b_ij(i, j, r);});
            }
            H(i-1, j-1) = h;
            B(i-1, j-1) = b;
        }
    }
    //std::cout << H << std::endl << std::endl << B << std::endl; // TODO RM
}


template<typename numeric_type>
numeric_type spherical_seq<numeric_type>::solution_n(unsigned int n, numeric_type r) {
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> coeffs = this->solution_coeffs.col(n);
    if (r == 0) {
        r = std::numeric_limits<numeric_type>::min();
    }
    numeric_type ret = 0;
    for (int i = 1; i < this->splines.num_knots - this->spline_order - 2; i++) {
        ret += coeffs(i) * this->splines.B_i(i, r);
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
        es(this->H, this->B);
    this-> energies = es.eigenvalues();
    this->solution_coeffs = es.eigenvectors();
    this->normalise_states();
    this->solved = true;
}

template<typename numeric_type>
void spherical_seq<numeric_type>::normalise_states() {
    //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT); // TODO RM
    for (int i = 0; i < this->solution_coeffs.cols(); i++) {
        numeric_type integral = 0.0;
        for (int j = 1; j < this->physical_points.size(); j++) {
            numeric_type dr = this->physical_points(j) - this->physical_points(j-1);
            numeric_type  r = this->physical_points(j-1) + dr/2;
            //
            integral += pow(r * solution_n(i, r),2) * dr;
        }
        this->solution_coeffs.col(i) /= integral;
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

template<typename numeric_type>
numeric_type spherical_seq<numeric_type>::H_B_i(unsigned int i, numeric_type r) {
    // TODO UNITS!
    numeric_type ret = 0.0;
    ret += - this->splines.B_i_xx(i, r) / 2;
    // prevent div by 0
    //ret += (this->l * (this->l + 1) / (2 * ((r > 1e-15) ? pow(r, 2) : pow(1e-15, 2))) + this->V(r)) * this->splines.B_i(i, r);
    ret += (r!= 0) ? (this->l * (this->l + 1) / (2 * (pow(r, 2))) + this->V(r)) * this->splines.B_i(i, r) : 0.0;

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
