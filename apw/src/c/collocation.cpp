#include "collocation.h"

collocation::collocation(const std::function<double(double)> &P,
                                       const std::function<double(double)> &Q,
                                       const std::function<double(double)> &G,
                                       const Eigen::VectorXd &physical_knots,
                                       const std::pair<double, double> &Boundary_Conditions):
p(P), q(Q), g(G), boundary_conditions(Boundary_Conditions) {

    interval = std::pair<double, double>( {physical_knots(0), physical_knots(Eigen::indexing::last)} );
    num_physical_points = physical_knots.size();

    // splines class
    int spline_order = 4; // always the same choice for 2nd order deqn
    splines = b_splines(spline_order, physical_knots);
    int num_knots = splines.num_knots; // = #phys + 6
    // coeff matrix dimension
    int num_unknowns = num_knots - spline_order; // = #phys + 2 = N - 4

    coeff_mat = Eigen::MatrixXd::Zero(num_unknowns, num_unknowns);
    rhs_vector = Eigen::VectorXd::Zero(num_unknowns);
    // this is not changed here but only after calling solve()
    solution_coeffs = Eigen::VectorXd::Zero(num_unknowns);

    // implement boundary conditions
    coeff_mat(0, 0) = double(1.0);
    rhs_vector(0) = boundary_conditions.first;
    coeff_mat(num_unknowns-1, num_unknowns-1) = double(1.0);
    rhs_vector(num_unknowns-1) = boundary_conditions.second;
    for (int i = 1; i < num_unknowns-1; i++) {

        int position_index = i - 1;
        double x = physical_knots(position_index);
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
    //std::cout << coeff_mat << std::endl; // Todo RM
}


collocation::collocation(const std::function<double(double)> &P,
                                       const std::function<double(double)> &Q,
                                       const std::function<double(double)> &G,
                                       const std::pair<double, double> &Interval,
                                       const std::pair<double, double> &Boundary_Conditions,
                                       int Num_Physical_Points) {
    // linspaced knots TODO not always a good choice, add option for self-specified points
    Eigen::VectorXd Physical_Knots =
        Eigen::VectorXd::LinSpaced(Num_Physical_Points, Interval.first, Interval.second);

    *this = collocation(P, Q, G, Physical_Knots, Boundary_Conditions);
}


collocation::collocation(const std::function<double(double)> &P,
                                       const std::function<double(double)> &Q,
                                       const std::function<double(double)> &G,
                                       const std::pair<double, double> &Interval,
                                       const std::pair<double, double> &Boundary_Conditions,
                                       const Eigen::VectorXd & Critical_Points,
                                       int Num_Physical_Points) {
    int num_critical_points = Critical_Points.size();
    double dx = (Interval.second - Interval.first) / (Num_Physical_Points - 1);
    double delta = double(1e-6); //TODO meh
    std::vector<double> linspace = std::vector<double>();

    int curr_crit_point = 0;
    double x = Interval.first;
    for (int i = 0; i < Num_Physical_Points; i++) {
        double next_x = x + dx;
        double critp = Critical_Points(curr_crit_point);
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

    Eigen::VectorXd Physical_Knots =
            Eigen::Map<Eigen::VectorXd>(linspace.data(), linspace.size());

    *this = collocation(P, Q, G, Physical_Knots, Boundary_Conditions);
}


void collocation::solve() {
    if (solved) {
        return;
    } else {
        Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXd>> LU(coeff_mat);
        solution_coeffs = LU.solve(rhs_vector);
        if (this->boundary_conditions.first == 0.0) solution_coeffs(0) = 0;
        if (this->boundary_conditions.second == 0.0) solution_coeffs(Eigen::indexing::last) = 0;
        solved = true;
    }
}


double collocation::solution(double x) {
    if (!solved) {
        throw std::runtime_error("Cannot access solution before calling solve()!");
    }
    double ret = 0;
    for (int i = 0; i < solution_coeffs.size(); i++) {
        ret += solution_coeffs(i) * splines.B_i(i, x);
    }

    return ret;
}


void
collocation::save_solution(double xmin, double xmax, int num_xs, const std::string &path) {
    Eigen::VectorXd xs =
            Eigen::VectorXd::LinSpaced(num_xs, xmin, xmax);
    Eigen::VectorXd fs(num_xs);
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