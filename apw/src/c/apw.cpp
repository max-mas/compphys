#include "apw.h"

apw::apw(unsigned int chargeNum, unsigned int lMax, double muffin_tin_radius, double k_cutoff, 
         std::vector<Vector3d> latticeVecs, double latticeConst, std::string path)
: lmax(lMax), Z(chargeNum), R(muffin_tin_radius), lattice_vecs(latticeVecs), lattice_constant(latticeConst) {
    cout << "--------------------------------------------------------------------------------------------------\n";
    cout << "Band structure calculation for Z = " << this->Z << ".\n";
    cout << "Lattice: \n";
    for (auto vec: this->lattice_vecs) cout << vec.transpose() << "; ";
    this->reciprocal_lattice_vecs = this->generate_reciprocal_lattice_basis(this->lattice_vecs);
    Vector3d & b1 = this->reciprocal_lattice_vecs[0];
    Vector3d & b2 = this->reciprocal_lattice_vecs[1];
    Vector3d & b3 = this->reciprocal_lattice_vecs[2];

    cout << "\nReciprocal lattice: \n";
    for (auto vec: this->reciprocal_lattice_vecs) cout << vec.transpose() << "; ";
    cout << endl; 
    cout << "--------------------\n";

    cout << "Solving atomic problem for 1+ ion to retrieve potential:" << endl; 
    this->ion = atom(chargeNum, chargeNum-1, 50, 50, 200, 500, 0.4, 5e-4); //TODO tolerance too big for final build
    // 50, 50, 200, 500, 0.4, 5e-4
    this->ion.run();
    cout << "--------------------" << endl;

    cout << "Generating finite set of K vectors to sum.";
    // use cutoff 3 or 4 for many vectors, 2 for testing
    int index_cutoff = 6;
    std::vector<Vector3d> Ks = this->generate_finite_reciprocal_lattice_set(k_cutoff, index_cutoff);
    size_t n_K = Ks.size();  
    cout << " Found " << n_K << " vectors with norm less than " << k_cutoff << " and coefficients less than " << index_cutoff << ".\n" << endl;
    
    //this->save_dets_high_symmetry(Ks, path);

    this->save_dets_BZ(Ks, path);

    

}

void apw::save_muffin_tin_orbital(int l, std::string path, std::vector<std::pair<std::function<double (double)>, std::function<double (double)>>> muffin_tin_functions) {
    VectorXd xs = VectorXd::LinSpaced(1000, 0.0, this->R + 1);

    std::ofstream file;
    file.open(path);
    // write
    for (int j = 0; j < xs.size(); j++) {
        file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
            << std::scientific << xs[j] << ","
            << std::scientific << muffin_tin_functions[l].first(xs[j]) << std::endl;
    }
    // don't forget to clean up :)
    file.close();
}

void apw::save_dets_high_symmetry(const std::vector<Vector3d> &Ks, std::string path) {
    cout << "-------------------------------------------------------------------------------------------------\n\n";
    cout << "Calculating determinant along k paths." << endl;    

    Vector3d G, H, P, N;
    G <<                        0,                     0,                     0;
    H <<  2*M_PI/lattice_constant,                     0,                     0;
    P <<    M_PI/lattice_constant, M_PI/lattice_constant, M_PI/lattice_constant;
    N <<    M_PI/lattice_constant, M_PI/lattice_constant,                     0;

    int k_steps = 100;
    int E_steps = 2000;
    cout << "Getting det from G to H point." << endl;
    Vector3d diff = H - G;
    std::string path1 = path + "g_h/";
    this->save_dets_on_k_path(k_steps, E_steps, G, diff, Ks, path1);

    cout << "Getting det from H to N point." << endl;
    diff = N - H;
    std::string path2 = path + "h_n/";
    this->save_dets_on_k_path(k_steps, E_steps, H, diff, Ks, path2);    

    cout << "Getting det from N to G point." << endl;
    diff = G - N;
    std::string path3 = path + "n_g/";
    this->save_dets_on_k_path(k_steps, E_steps, N, diff, Ks, path3);

    cout << "Getting det from G to P point." << endl;
    diff = P - G;
    std::string path4 = path + "g_p/";
    this->save_dets_on_k_path(k_steps, E_steps, G, diff, Ks, path4);

    cout << "Getting det from P to H point." << endl;
    diff = H - P;
    std::string path5 = path + "p_h/";
    this->save_dets_on_k_path(k_steps, E_steps, P, diff, Ks, path5);
}

void apw::save_dets_BZ(const std::vector<Vector3d> &Ks, std::string path) {
    cout << "-------------------------------------------------------------------------------------------------\n\n";
    cout << "Calculating determinant in BZ for DOS estimation." << endl;    

    size_t n_K = Ks.size();
    size_t E_steps = 2000;

    int N = 20; // 10^3 = 1000 BZ pts
    VectorXd mpack_coeffs = this->monkhorst_pack_factors(N);

    progressbar bar(N * N * N);

    std::filesystem::create_directory(path);

    for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
    for (int l = 0; l < N; l++) {
        Vector3d k_vec =  mpack_coeffs(i) * this->reciprocal_lattice_vecs[0]
                        + mpack_coeffs(j) * this->reciprocal_lattice_vecs[1]
                        + mpack_coeffs(l) * this->reciprocal_lattice_vecs[2];

        double E_APW = k_vec.norm() * k_vec.norm() / 2;
        std::vector<std::pair<std::function<double (double)>, std::function<double (double)>>> 
            muffin_tin_functions =  this->get_muffin_tin_functions_E(E_APW);


        std::vector<double> Es;
        std::vector<double> dets;

        MatrixXd H = MatrixXd::Zero(n_K, n_K);
        VectorXd test_Es = VectorXd::LinSpaced(E_steps, -1.0, 1.0);
        for (double E : test_Es) {       
            this->generate_H(H, E, k_vec, Ks, muffin_tin_functions);  // update H in place             
            double det = (H - E * MatrixXd::Identity(n_K, n_K)).determinant();
            Es.emplace_back(E);
            dets.emplace_back(det);
        }

        std::ofstream file;
        file.open(path  + "det_" + std::to_string(k_vec(0)) + "_" + std::to_string(k_vec(1)) 
                        + "_" + std::to_string(k_vec(2)) + ".txt");
        // write
        for (int lineindex = 0; lineindex < dets.size(); lineindex++) {
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
                << std::scientific << Es[lineindex] << ","
                << std::scientific << dets[lineindex] << std::endl;
        }
        // don't forget to clean up :)
        file.close();

        
        bar.update();
    }
    }
    }
    

}

void apw::save_dets_on_k_path(int k_steps, int E_steps, Vector3d k0, Vector3d k_diff, std::vector<Vector3d> Ks, std::string path) {
    size_t n_K = Ks.size();
    VectorXd k_scaling_factos = VectorXd::LinSpaced(k_steps, DBL_EPSILON, 1.0);

    progressbar bar(k_steps);
    for (double k : k_scaling_factos) {
        Vector3d k_vec;
        k_vec = k0 + k * k_diff;
        double E_APW = k_vec.norm() * k_vec.norm() / 2; // test value   
        std::vector<std::pair<std::function<double (double)>, std::function<double (double)>>> 
            muffin_tin_functions =  this->get_muffin_tin_functions_E(E_APW);

        //for (int l = 0; l <= this->lmax; l++) this->save_muffin_tin_orbital(l, "../results/muffintin_k" 
        //    + std::to_string(k) + "_l" + std::to_string(l) + ".txt", muffin_tin_functions); // for testing

        std::vector<double> Es;
        std::vector<double> dets;

        MatrixXd H = MatrixXd::Zero(n_K, n_K);
        VectorXd test_Es = VectorXd::LinSpaced(E_steps, -1.0, 1.0);
        for (double E : test_Es) {       
            this->generate_H(H, E, k_vec, Ks, muffin_tin_functions);  // update H in place             
            double det = (H - E * MatrixXd::Identity(n_K, n_K)).determinant();
            Es.emplace_back(E);
            dets.emplace_back(det);
        }

        std::ofstream file;
        file.open(path + "det_k" + std::to_string(k) + ".txt");
        // write
        for (int j = 0; j < dets.size(); j++) {
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1) // get all digits
                << std::scientific << Es[j] << ","
                << std::scientific << dets[j] << std::endl;
        }
        // don't forget to clean up :)
        file.close();
        
        bar.update();
    }
    cout << endl;
}

std::vector<std::pair<std::function<double (double)>, std::function<double (double)>>>
    apw::get_muffin_tin_functions_E(double E) {
    std::vector<std::pair<std::function<double (double)>, std::function<double (double)>>> muffin_tin_functions;

    for (int l = 0; l <= this->lmax; l++) {
        // 2nd order ode (SEQ) rhs
        auto seq = [&](const Vector2d & x, Vector2d & dx, double r){
            if (abs(r) < DBL_EPSILON) r = DBL_EPSILON; 
            dx(0) = x(1);
            dx(1) = (l*(l + 1)/(r * r) + 2*this->ion.mean_field_potential(r) - 2*E) * x(0);
        };

        // record numerical integration
        struct observe {
            std::vector<double> & states;
            std::vector<double>& positions;

            observe(std::vector<double> & states_init, std::vector<double> & positions_init )
            : states( states_init ) , positions( positions_init ) { }

            void operator() (const Vector2d &x , double r) {
                states.emplace_back(x(0));
                positions.emplace_back(r);
            }
        };
        // containers for P(x)
        std::vector<double> states;
        std::vector<double> positions;

        observe observer(states, positions);
        
        Vector2d state;
        double stepsize = 5e-3; // stepsize
        state << 0.0, pow(stepsize, l); // initial conditions
        // integrate
        boost::numeric::odeint::runge_kutta4<Vector2d> stepper;
        int steps = boost::numeric::odeint::integrate_const(stepper, seq, state, 0.0, this->R+1, stepsize, observer); // integrate further than stepsize to improve accuracy of derivative at R
        double scaling_factor = *std::max_element(states.begin(), states.end());

        cubic_b_spline<double> spline(states.begin(), states.end(), 0.0, stepsize);
        auto orbital = [spline, scaling_factor](double r){
            if (r < DBL_EPSILON) r = DBL_EPSILON; 
            return spline(r) / (r * scaling_factor) ;}; // 

        auto orbital_derivative = [spline, scaling_factor](double r){
            if (r < DBL_EPSILON) r = DBL_EPSILON; 
            return (spline.prime(r) / r - spline(r) / pow(r, 2)) / scaling_factor ;}; // 

        muffin_tin_functions.emplace_back(
            std::pair<std::function<double (double)>, std::function<double (double)>>{orbital, orbital_derivative});
    }

    return muffin_tin_functions;
}

complex<double> apw::psi_apw(Vector3d q, Vector3d r, cubic_b_spline<double> muffintin) {
    Vector3d q_sph = this->cartesian_to_spherical(q);
    Vector3d r_sph = this->cartesian_to_spherical(r);
    if (r_sph(0) >= this->R) return this->truncated_plane_wave(q, r);

    complex<double> ret = 0.0;
    for (int l = 0; l <= this->lmax; l++) {
        for (int m = -l; m <= l; m++) {
            ret += std::pow(complex<double>(0, 1), l) * sph_bessel(l, q_sph(0) * r_sph(0))
                / muffintin(this->R) * muffintin(r_sph(0))
                * conj(spherical_harmonic(l, m, q_sph(1), q_sph(2)))
                * (spherical_harmonic(l, m, r_sph(1), r_sph(2)));
        }
    }
    return 4.0 * M_PI * ret;
}

complex<double> apw::truncated_plane_wave(Vector3d q, Vector3d r)
{
    Vector3d q_sph = this->cartesian_to_spherical(q);
    Vector3d r_sph = this->cartesian_to_spherical(r);

    complex<double> ret = 0.0;
    for (int l = 0; l <= this->lmax; l++) {
        for (int m = -l; m <= l; m++) {
            ret += std::pow(complex<double>(0, 1), l) * sph_bessel(l, q_sph(0) * r_sph(0))
                * conj(spherical_harmonic(l, m, q_sph(1), q_sph(2)))
                * (spherical_harmonic(l, m, r_sph(1), r_sph(2)));
        }
    }
    return 4.0 * M_PI * ret;
}

Vector3d apw::cartesian_to_spherical(Vector3d x) {
    double r = sqrt(pow(x(0), 2) + pow(x(1), 2) + pow(x(2), 2));
    double theta = atan2( sqrt( pow(x(0), 2) + pow(x(1), 2) ), x(2));
    double phi = atan2(x(1), x(0));

    Vector3d ret;
    ret << r, theta, phi;

    return ret;
}

std::vector<Vector3d> apw::generate_reciprocal_lattice_basis(std::vector<Vector3d> lattice) {
    Vector3d & A = lattice[0];
    Vector3d & B = lattice[1];
    Vector3d & C = lattice[2];
    double V = A.transpose() * (B.cross(C)); 
    cout << "Unit cell volume: " << V << endl;

    Vector3d B1 = 2*M_PI / V * B.cross(C);
    Vector3d B2 = 2*M_PI / V * C.cross(A);
    Vector3d B3 = 2*M_PI / V * A.cross(B);

    return std::vector<Vector3d>({B1, B2, B3});
}

std::vector<Vector3d> apw::generate_finite_reciprocal_lattice_set(double norm_cutoff, int index_cutoff) {
    Vector3d & b1 = this->reciprocal_lattice_vecs[0];
    Vector3d & b2 = this->reciprocal_lattice_vecs[1];
    Vector3d & b3 = this->reciprocal_lattice_vecs[2];

    std::vector<Vector3d> ret;

    for (int n = -index_cutoff; n<= index_cutoff; n++) {
        for (int m = -index_cutoff; m<= index_cutoff; m++) {
            for (int l = -index_cutoff; l<= index_cutoff; l++) {
                Vector3d K = n * b1 + m * b2 + l * b3;
                if (K.norm() <= norm_cutoff) ret.emplace_back(K);
            }
        }
    }

    return ret;
}

void apw::generate_H(MatrixXd & H, double E, Vector3d k, std::vector<Vector3d> finite_reciprocal_lattice_set,
                         std::vector<std::pair<std::function<double (double)>, std::function<double (double)>>>  muffin_tin_functions) {
    size_t n_K = finite_reciprocal_lattice_set.size(); 
    double V = abs(this->lattice_vecs[0].transpose() * 
        (this->lattice_vecs[1].cross(this->lattice_vecs[2]))); // TODO factor of 2, test

    for (int i = 0; i < n_K; i++) {
#pragma omp parallel for // maybe this helps speed up matrix creation?
        for (int j = 0; j < n_K; j++) {
            Vector3d & K_i = finite_reciprocal_lattice_set[i];
            Vector3d & K_j = finite_reciprocal_lattice_set[j];
            Vector3d q_i = k + K_i;
            Vector3d q_j = k + K_j;
            double diff = (K_i - K_j).norm();

            double R2 = pow(this->R, 2);
            double quot = (diff == 0.0) ? this->R/3 : sph_bessel(1, diff * this->R) / diff;
            double kronecker = (i==j) ? 1.0 : 0.0;
            double A_ij = -4.0 * M_PI * R2 / V * quot + kronecker;
            
            double dotprod = q_i.dot(q_j);
            double B_ij = 0.5 * A_ij * dotprod;

            double qi_norm = q_i.norm();
            double qj_norm = q_j.norm();
            double C_ij = 0.0;
            for (int l = 0; l < this->lmax; l++) {
                double legendre_arg = dotprod / (qi_norm * qj_norm);
                if (legendre_arg > 1) legendre_arg = 1.0;
                if (legendre_arg < -1) legendre_arg = -1.0;
                C_ij += (2*l + 1) * 2 * M_PI * R2 / V 
                    * legendre_p(l, legendre_arg) 
                    * sph_bessel(l, qi_norm * this->R) * sph_bessel(l, qj_norm * this->R)
                    * muffin_tin_functions[l].second(this->R) // derivative
                    / muffin_tin_functions[l].first(this->R);
            }        
            double matel = - E * A_ij + B_ij + C_ij;            
            H(i, j) = matel;
        }
    }
}

VectorXd apw::monkhorst_pack_factors(int N) {
    VectorXd ret = VectorXd::Zero(N);

    for (int r = 1; r <= N; r++) {
        ret(r-1) = double((2 * r - N - 1)) / double((2*N));
    }

    return ret;
}
