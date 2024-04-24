//
// Created by max on 4/20/24.
//

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
 * @tparam numeric_type Numeric type like float, double that has an
 *                      order relation "<" (this generally rules out complex numbers!)
 */
template <typename numeric_type>
class collocation {
public:
    // prefactor of the 1st derivative
    std::function<numeric_type (numeric_type)> p;
    // prefactor of the linear term
    std::function<numeric_type (numeric_type)> q;
    // RHS
    std::function<numeric_type (numeric_type)> g;
    // solution interval
    std::pair<numeric_type, numeric_type> interval;
    // boundary conditions, here: f on the boundaries of the interval
    std::pair<numeric_type, numeric_type> boundary_conditions;
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
    collocation(const std::function<numeric_type (numeric_type)> & P,
                const std::function<numeric_type (numeric_type)> & Q,
                const std::function<numeric_type (numeric_type)> & G,
                const Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> & physical_knots,
                const std::pair<numeric_type, numeric_type> & Boundary_Conditions);

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
    collocation(const std::function<numeric_type (numeric_type)> & P,
                const std::function<numeric_type (numeric_type)> & Q,
                const std::function<numeric_type (numeric_type)> & G,
                const std::pair<numeric_type, numeric_type> & Interval,
                const std::pair<numeric_type, numeric_type> & Boundary_Conditions,
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
    collocation(const std::function<numeric_type (numeric_type)> & P,
                const std::function<numeric_type (numeric_type)> & Q,
                const std::function<numeric_type (numeric_type)> & G,
                const std::pair<numeric_type, numeric_type> & Interval,
                const std::pair<numeric_type, numeric_type> & Boundary_Conditions,
                const Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> & Critical_Points,
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
    numeric_type solution(numeric_type x);

    /**
     * Saves the previously calculated solution on a discrete sample interval.
     * @param xmin
     * @param xmax
     * @param num_xs
     * @param path Path to a txt file or equivalent
     */
    void save_solution(numeric_type xmin, numeric_type xmax, int num_xs, const std::string & path);

private:
    // if false, solution() and save_solution() are inaccessible / throw exceptions
    bool solved = false;

    // b spline object of order 4
    b_splines<numeric_type> splines;

    // coefficient matrix, is destroyed in place by solve()!
    Eigen::Matrix<numeric_type, Eigen::Dynamic, Eigen::Dynamic> coeff_mat;
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> rhs_vector;
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> solution_coeffs;

};

template<typename numeric_type>
collocation<numeric_type>::collocation(const std::function<numeric_type(numeric_type)> &P,
                                       const std::function<numeric_type(numeric_type)> &Q,
                                       const std::function<numeric_type(numeric_type)> &G,
                                       const Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> &physical_knots,
                                       const std::pair<numeric_type, numeric_type> &Boundary_Conditions):
p(P), q(Q), g(G), boundary_conditions(Boundary_Conditions) {

    interval = std::pair<numeric_type, numeric_type>( {physical_knots(0), physical_knots(Eigen::indexing::last)} );
    num_physical_points = physical_knots.size();

    // splines class
    int spline_order = 4; // always the same choice for 2nd order deqn
    splines = b_splines<numeric_type>(spline_order, physical_knots);
    int num_knots = splines.num_knots; // = #phys + 6
    // coeff matrix dimension
    int num_unknowns = num_knots - spline_order; // = #phys + 2 = N - 4

    coeff_mat = Eigen::Matrix<numeric_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(num_unknowns, num_unknowns);
    rhs_vector = Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>::Zero(num_unknowns);
    // this is not changed here but only after calling solve()
    solution_coeffs = Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>::Zero(num_unknowns);

    // implement boundary conditions
    coeff_mat(0, 0) = numeric_type(1.0);
    rhs_vector(0) = boundary_conditions.first;
    coeff_mat(num_unknowns-1, num_unknowns-1) = numeric_type(1.0);
    rhs_vector(num_unknowns-1) = boundary_conditions.second;
    for (int i = 1; i < num_unknowns-1; i++) {

        int position_index = i - 1;
        numeric_type x = physical_knots(position_index);
        int min_nonzero_spline = i-1;
        int max_nonzero_spline = i + spline_order - 3;
        if (i == num_unknowns-2) {x -= 1e-12;} // TODO giga annoying workaround
        for (int n = min_nonzero_spline; n <= max_nonzero_spline; n++) {
            if (n >= num_unknowns) {
                continue;
            } else {
                coeff_mat(i, n) = splines.B_i_xx(n, x) + p(x) * splines.B_i_x(n, x) + q(x) * splines.B_i(n, x);
            }
        }

        rhs_vector(i) = g(x);
    }
    std::cout << coeff_mat << std::endl; // Todo RM
}

template<typename numeric_type>
collocation<numeric_type>::collocation(const std::function<numeric_type(numeric_type)> &P,
                                       const std::function<numeric_type(numeric_type)> &Q,
                                       const std::function<numeric_type(numeric_type)> &G,
                                       const std::pair<numeric_type, numeric_type> &Interval,
                                       const std::pair<numeric_type, numeric_type> &Boundary_Conditions,
                                       int Num_Physical_Points) {
    // linspaced knots TODO not always a good choice, add option for self-specified points
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> Physical_Knots =
        Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>::LinSpaced(Num_Physical_Points, Interval.first, Interval.second);

    *this = collocation<numeric_type>(P, Q, G, Physical_Knots, Boundary_Conditions);
}

template<typename numeric_type>
collocation<numeric_type>::collocation(const std::function<numeric_type(numeric_type)> &P,
                                       const std::function<numeric_type(numeric_type)> &Q,
                                       const std::function<numeric_type(numeric_type)> &G,
                                       const std::pair<numeric_type, numeric_type> &Interval,
                                       const std::pair<numeric_type, numeric_type> &Boundary_Conditions,
                                       const Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> & Critical_Points,
                                       int Num_Physical_Points) {
    int num_critical_points = Critical_Points.size();
    numeric_type dx = (Interval.second - Interval.first) / (Num_Physical_Points - 1);
    numeric_type delta = numeric_type(1e-6); //TODO meh
    std::vector<numeric_type> linspace = std::vector<numeric_type>();

    int curr_crit_point = 0;
    numeric_type x = Interval.first;
    for (int i = 0; i < Num_Physical_Points; i++) {
        numeric_type next_x = x + dx;
        numeric_type critp = Critical_Points(curr_crit_point);
        if (x < critp and next_x > critp and abs(next_x-critp) > delta) {
            linspace.emplace_back(x);
            linspace.emplace_back(critp - delta);
            linspace.emplace_back(critp);
            linspace.emplace_back(critp + delta);
            if (num_critical_points > curr_crit_point+1) {curr_crit_point++;}
        } else if (x < critp and abs(next_x-critp) < delta) {
            linspace.emplace_back(x);
            linspace.emplace_back(critp - delta);
            linspace.emplace_back(critp);
            linspace.emplace_back(critp + delta);
            x += dx;
            if (num_critical_points > curr_crit_point+1) {curr_crit_point++;}
        } else {
            linspace.emplace_back(x);
        }
        x += dx;
    }

    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> Physical_Knots =
            Eigen::Map<Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>>(linspace.data(), linspace.size());

    *this = collocation<numeric_type>(P, Q, G, Physical_Knots, Boundary_Conditions);
}

template<typename numeric_type>
void collocation<numeric_type>::solve() {
    if (solved) {
        return;
    } else {
        Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXd>> LU(coeff_mat);
        solution_coeffs = LU.solve(rhs_vector);
        solved = true;
    }
}

template<typename numeric_type>
numeric_type collocation<numeric_type>::solution(numeric_type x) {
    if (!solved) {
        throw std::runtime_error("Cannot access solution before calling solve()!");
    }
    numeric_type ret = 0;
    for (int i = 0; i < solution_coeffs.size(); i++) {
        ret += solution_coeffs(i) * splines.B_i(i, x);
    }

    return ret;
}

template<typename numeric_type>
void
collocation<numeric_type>::save_solution(numeric_type xmin, numeric_type xmax, int num_xs, const std::string &path) {
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> xs =
            Eigen::Matrix<numeric_type, Eigen::Dynamic, 1>::LinSpaced(num_xs, xmin, xmax);
    Eigen::Matrix<numeric_type, Eigen::Dynamic, 1> fs(num_xs);
    for (int i = 0; i < num_xs; i++) {
        fs(i) = solution(xs(i));
    }

    std::ofstream file;
    file.open(path);
    // write
    for (int j = 0; j < num_xs; j++) {
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
             << std::scientific << xs[j] << ","
             << std::scientific << fs[j] << std::endl;
    }
    // don't forget to clean up :)
    file.close();

}


#endif //POISSON_COLLOCATION_H
