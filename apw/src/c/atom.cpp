#include "atom.h"

atom::atom(int chargeNumber, int electronNumber, int maxIt,
                         double rMax, int numKnotsSeq, int numKnotsPot,
                         double Eta, double tolerance, Eigen::MatrixXi occupationInit):
Z(chargeNumber), N_e(electronNumber), max_it_self_consistency(maxIt), r_max(rMax),
num_knots_seq(numKnotsSeq), num_knots_pot(numKnotsPot), eta(Eta), tol(tolerance) {
    // check if occupation prescribed:
    if (!(occupationInit.rows() == 1 and occupationInit(0, 0) == 0)) {
        this->assign_orbitals_auto = false;
        this->occupation_matrix = occupationInit;
        this->orig_occupation_matrix = occupationInit;
        std::cout << "Using pre-assigned orbital configuration!" << std::endl;
    }

    // Generate knot sequences
    this->knots_seq = generate_knots(this->num_knots_seq, this->r_max);
    this->knots_pot = generate_knots(this->num_knots_pot, this->r_max);

    // Needed to prevent segfault
    this->seq_solvers = std::vector<spherical_seq>();

    // Potentially needed to prevent segfault
    this->potential_solver = collocation();

    // Solve SEQ for all l
    std::cout << "Find atomic structure for Z = " << this->Z << " and " << this->N_e << " electrons." << std::endl;
    std::cout << "Initial SEQ solution with V = 0." << std::endl;
    for (int l = 0; l <= this->l_max; l++) {
        std::cout << "l = " << l << ", " << std::flush;
        auto fn = [&](double r){return this->mean_field_potential(r);};
        this->seq_solvers.emplace_back(spherical_seq(this->knots_seq, fn, l, 4));
        this->seq_solvers[l].solve();
    }
    std::cout << std::endl;

    // Assign electrons
    this->occupation_matrix = this->determine_occupation();
    // Save for convergence checking
    this->curr_erg = this->total_energy();
    // Print total energy
    std::cout << "Total energy: " << this->total_energy() << std::endl;

    // Solve poisson eqn
    auto zero_fn = [](double r){return 0.0;};
    this->potential_solver = collocation(zero_fn, zero_fn,
                                                       [&](double r){return this->colloc_rhs(r);},
                                                       this->knots_pot, {0.0, N_e});
    this->potential_solver.solve();
    std::cout << "Initialised rho solution..." << std::endl;
    // Sanity check
    this->check_rho();
    // Set init flag
    this->init_finished = true;
}


void atom::run() {
    // Throw a fit if run() has been called before
    if (this->ran) throw std::runtime_error("Computation has been run before. Create a new object.");

    // Tolerance for convergence check of 1s energy TODO what would a sensible value be? rn 4 decimals
    for (int i = 1; i <= this->max_it_self_consistency; i++) {
        std::cout << "Iteration " << i << ":" << std::endl;
        this->iteration(i); // perform an iteration
        double change_erg = abs(this->curr_erg - this->prev_erg);
        if (change_erg < this->tol) { // convergence check
            std::cout << "Tolerance of " << this->tol << " reached after " << i << " iterations. Stopping." << std::endl;
            this->ran = true; // flag
            return;
        } 
        std::cout << "Total energy delta: " << change_erg << std::endl;
    }
    // If not converged:
    std::cout << "Tolerance of " << this->tol << " not reached after " << this->max_it_self_consistency
        << " iterations. Stopping." << std::endl;    

    this->ran = true; // flag
}


void atom::iteration(int i) {
    //TODO
    this->previous_seq_solvers = this->seq_solvers;
    this->prev_occupation_matrix = this->occupation_matrix;
    // Solve SEQ for all l
    for (int l = 0; l <= this->l_max; l++) {
        std::cout << "l = " << l << ", " << std::flush;
        this->seq_solvers[l] = spherical_seq(this->knots_seq,
                                                [&](double r){return this->mixed_mean_field_potential(r);}, l, 4);
        this->seq_solvers[l].solve();
    }
    std::cout << std::endl;
    if (! this->assign_orbitals_auto) this->occupation_matrix = this->orig_occupation_matrix;

    // Assign electrons to orbitals
    this->occupation_matrix = this->determine_occupation();
    // For convergence checking
    double e = this->total_energy();
    this->prev_erg = this->curr_erg;
    this->curr_erg = e;
    // Print total energy
    std::cout << "Total energy: " << e << std::endl;

    // Solve Poisson's eqn
    this->previous_potential_solver = this->potential_solver;
    auto zero_fn = [](double r){return 0.0;};
    this->potential_solver = collocation(zero_fn, zero_fn,
                                                       [&](double r){return this->colloc_rhs(r);},
                                                       this->knots_pot, {0.0, N_e});
    this->potential_solver.solve();
    std::cout << "Updated rho solution..." << std::endl;
    // Sanity check
    this->check_rho();
    if (i == 1) this->second_it_finished = true; // flag
}

// row index l, column index n

Eigen::MatrixXi atom::determine_occupation() {
    Eigen::MatrixXi retmat = (this->assign_orbitals_auto) ? 
        Eigen::MatrixXi::Zero(this->l_max + 1, this->knots_seq.size()) : this->occupation_matrix;
    // key: E, value: l, n (n-1 here!)
    this->bound_states = std::map<double, std::pair<int, int>>();
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            double E = this->seq_solvers[l].energies(n);
            if (E < 0) {
                this->bound_states[E] = {l, n+l}; // this n corresponds to n - 1  !!!
            } else {
                break;
            }
        }
    }
    // assign electrons to minimise energy until all assigned
    // don't do this if configuration prescribed!
    int unassigned_electrons = this-> N_e;
    if (this->assign_orbitals_auto) for (const auto& [E, qnums] : this->bound_states) {
        if (unassigned_electrons > 0) {
            int l = qnums.first;
            int n = qnums.second;
            int multiplicity = 2 * (2 * l + 1);

            for (int it = 0; it < multiplicity; it++) {
                if (unassigned_electrons > 0) {
                    retmat(l, n) += 1;
                    unassigned_electrons -= 1;
                } else break;
            }
        } else break;
    }
    // Print summary of occupancy and orbital energies
    static bool problem = false;
    bool tried_to_solve_problem = false;
    std::cout << "Occupancy:\n";
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            double E = seq_solvers[l].energies(n-l);
            if (retmat(l,n) > 0) std::cout << n+1 << this->orbital_names[l] << ": "
                << retmat(l,n) <<", E = " << E << "\n"; // n - l !
            // fucked up convergence shit for predetermined orbitals :(
            if (E > 0 and retmat(l,n) > 0) {
                std::cout << "W A R N I N G: State " << n+1 << this->orbital_names[l] << " has a positive energy."   << std::endl;
                if (!problem) {
                    this->seq_solvers[l] = this->previous_seq_solvers[l];  
                    problem = true;
                    tried_to_solve_problem = true;
                    std::cout << "Replacing with previous solution to help convergence (?)." << std::endl;
                    continue;
                } else {
                    retmat(l, n) = 0;
                    std::cout << "De-occupying to help convergence (?)." << std::endl;
                }                
            }
        }
    }
    if (!tried_to_solve_problem) problem = false;

    return retmat;
}


double atom::total_energy() {
    double ret = 0.0;
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            int occ = this->occupation_matrix(l, n);
            // skip integration for unoccupied / invalid states
            if (occ < 1) continue; else if (l > n) break;

            // V expectation value integral
            boost::math::quadrature::exp_sinh<double> integrator;
            auto f = [&](double r)
                    {return pow(r * this->seq_solvers[l].solution_n(n-l, r), 2) * this->mixed_many_body_potential(r);};
            double V_int = integrator.integrate(f);
            //
            ret += occ * (this->seq_solvers[l].energies(n-l) - 0.5 * V_int); // n - l !
        }
    }
    return ret;
}


void atom::check_rho() {
    // Integrate rho and print to see if result equals N_e
    boost::math::quadrature::exp_sinh<double> integrator; // domain [0, inf)
    auto f = [&](double r) {  return pow(r, 2) * this->rho(r) ;};
    double N = 4.0 * M_PI *  integrator.integrate(f);
    std::cout << "Numerical N_e: " << N << ", expected: " << this->N_e << std::endl;
}


double atom::rho(double r) {
    if (r <= 0) r = std::numeric_limits<double>::epsilon();
    return this->colloc_rhs(r) / (-r * 4.0 * M_PI) ;
}


double atom::prev_rho(double r) {
    if (r <= 0) r = std::numeric_limits<double>::epsilon();
    return this->prev_colloc_rhs(r) / (-r * 4.0 * M_PI) ;
}


Eigen::VectorXd atom::generate_knots(int n, double R) {
    //return Eigen::VectorXd::LinSpaced(n, 0.0, R);
    Eigen::VectorXd points = Eigen::VectorXd::Zero(n);
    double delta = 0.06;
    double rp = R / (exp((n-1) * delta) - 1);
    for (int i = 1; i < points.size(); i++) {
        points(i) = rp * (exp(i * delta) - 1);
    }

    /*double exponent = -6;
    double end_exponent = log10(R);
    double exponent_step = (end_exponent - exponent) / (n - 2);
    for (int i = 1; i < points.size(); i++) {
        points(i) = pow(10, exponent);
        exponent += exponent_step;
    }*/

    return points;
}


double atom::electrostatic_potential(double r) {
    if (r <= 0.0) r = std::numeric_limits<double>::epsilon();
    //if (r <= 0.0) r = 1e-8;
    return (this->init_finished) ? this->potential_solver.solution(r)/r : 0.0;
}


double atom::exchange_potential(double r) {
    if (r <= 0.0) r = std::numeric_limits<double>::epsilon();
    // 4 pi epsilon = e = 1
    return (this->init_finished) ?
        -3.0 * pow(3.0 * abs(this->rho(r)) / (8.0 * double(M_PI)), 1.0/3.0) : 0.0;
}


double atom::many_body_potential(double r) {
    // e = 1
    return (this->N_e > 1) ? 1.0 * this->electrostatic_potential(r) + 1.0 * this->exchange_potential(r) : 0.0;
}


double atom::previous_electrostatic_potential(double r) {
    if (r <= 0.0) r = std::numeric_limits<double>::epsilon();
    //if (r <= 0.0) r = 1e-8;
    return (this->second_it_finished) ? this->previous_potential_solver.solution(r)/r : 0.0;
}


double atom::previous_exchange_potential(double r) {
    if (r <= 0.0) r = std::numeric_limits<double>::epsilon();
    // 4 pi epsilon = e = 1
    return (this->second_it_finished) ?
        -3.0 * pow(3.0 * abs(this->prev_rho(r)) / (8.0 * double(M_PI)), 1.0/3.0) : 0.0;
}


double atom::previous_many_body_potential(double r) {
    // e = 1
    return (this->N_e > 1) ?
        1.0 * this->previous_electrostatic_potential(r) + 1.0 * this->previous_exchange_potential(r) : 0.0;
}


double atom::nuclear_potential(double r) {
    return -double(this->Z) / r;
}


double atom::mean_field_potential(double r) {
    return this->nuclear_potential(r) + this->many_body_potential(r);
}


double atom::mixed_mean_field_potential(double r) {
    return this->nuclear_potential(r) + (1 - this->eta) * this->many_body_potential(r)
                                      +      this->eta  * this->previous_many_body_potential(r);
}


double atom::mixed_many_body_potential(double r) {
    return (1 - this->eta) * this->many_body_potential(r) + this->eta * this->previous_many_body_potential(r);
}


double atom::zero_function(double r) {
    return 0.0;
}


double atom::colloc_rhs(double r) {
    double ret = 0.0;
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            int occ = this->occupation_matrix(l, n);
            ret += (occ > 0) ? 1.0 * occ * pow(this->seq_solvers[l].solution_n(n-l, r), 2) : 0.0; // - <=> e??
        }
    }
    return -r * ret;
}


double atom::prev_colloc_rhs(double r) {
    double ret = 0.0;
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            int occ = this->prev_occupation_matrix(l, n);
            ret += (occ > 0) ? 1.0 * occ * pow(this->previous_seq_solvers[l].solution_n(n-l, r), 2) : 0.0; // - <=> e??
        }
    }
    return -r * ret;
}


void atom::save_summary(const std::string& path) {
    std::ofstream file;
    file.open(path);
    for (int l = 0; l <= this-> l_max; l++) {
        for (int n = 0; n < this->knots_seq.size(); n++) {
            if (this->occupation_matrix(l,n) > 0) {
                file << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                     << n + 1 << this->orbital_names[l] << ": "
                     << this->occupation_matrix(l, n) << ", " << std::scientific << this->seq_solvers[l].energies(n - l)
                     << std::endl; // n - l !
            }
        }
    }
    file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific
         << this->total_energy() << std::endl;
    // don't forget to clean up :)
    file.close();
}


void atom::save_potentials(const std::string& path) {
    std::ofstream file1;
    std::ofstream file2;

    int num_pts = 3000;
    auto points = Eigen::VectorXd::LinSpaced(num_pts, 0.0, this->r_max);

    file1.open(path + "direct_ne_" + std::to_string(this->N_e) + ".txt");
    file2.open(path + "exchange_ne_" + std::to_string(this->N_e) + ".txt");
    for (auto x : points) {
        file1 << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
              << std::scientific << x << "," << std::scientific << this->electrostatic_potential(x) << std::endl;

        file2 << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
              << std::scientific << x << "," << std::scientific << this->exchange_potential(x) << std::endl;
    }
    // don't forget to clean up :)
    file1.close();
    file2.close();
}


void atom::save_rho(const std::string& path) {
    std::ofstream file;

    int num_pts = 3000;
    auto points = Eigen::VectorXd::LinSpaced(num_pts, 0.0, this->r_max);

    file.open(path);
    for (auto x : points) {
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
              << std::scientific << x << "," << std::scientific << this->rho(x) << std::endl;
    }
    // don't forget to clean up :)
    file.close();
}