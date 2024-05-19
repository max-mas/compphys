#include "apw.h"

apw::apw(unsigned int chargeNum, unsigned int lMax, double muffin_tin_radius, 
         std::vector<Vector3d> latticeVecs, double latticeConst)
: lmax(lMax), Z(chargeNum), R(muffin_tin_radius), lattice_vecs(latticeVecs), lattice_constant(latticeConst) {
    cout << "--------------------------------------------------------------------------------------------------\n";
    cout << "Band structure calculation for Z = " << this->Z << ".\n";
    cout << "Lattice: \n";
    for (auto vec: this->lattice_vecs) cout << vec.transpose() << "; ";
    this->reciprocal_lattice_vecs = this->generate_reciprocal_lattice(this->lattice_vecs);
    cout << "\nReciprocal lattice: \n";
    for (auto vec: this->reciprocal_lattice_vecs) cout << vec.transpose() << "; ";
    cout << endl; 
    cout << "--------------------\n";

    cout << "Solving atomic problem for 1+ ion to retrieve potential:" << endl; 
    this->ion = atom(chargeNum, chargeNum-1, 50, 50, 200, 500, 0.4, 5e-2); //TODO tolerance too big for final build
    this->ion.run();
    cout << "--------------------" << endl;

    Vector3d k;
    k << 2*M_PI / latticeConst * 0.0, 0.0, 0.0;
    double E = k.norm() * k.norm() / 2; // test value
    cout << "Test energy: " << E << endl;

    for (int l = 0; l <= lMax; l++) {
        auto seq = [&](const Vector2d & x, Vector2d & dx, double r){
            if (abs(r) < DBL_EPSILON) r = DBL_EPSILON; 
            dx(0) = x(1);
            dx(1) = -(l*(l + 1)/(r * r) + 2*this->ion.mean_field_potential(r) - 2*E) * x(0);
        };

        struct observe {
            std::vector<Vector2d> & states;
            std::vector<double>& positions;

            observe(std::vector<Vector2d> & states_init, std::vector<double> & positions_init )
            : states( states_init ) , positions( positions_init ) { }

            void operator() (const Vector2d &x , double r) {
                states.push_back(x);
                positions.push_back(r);
            }
        };
        std::vector<Vector2d> states;
        std::vector<double> positions;

        observe observer(states, positions);
        
        Vector2d state;
        double stepsize = 1e-2;
        state << 0.0, 1.0;//pow(stepsize, l);
        boost::numeric::odeint::runge_kutta4<Vector2d> stepper;
        int steps = boost::numeric::odeint::integrate_const(stepper, seq, state, 0.0, this->R, stepsize, observer);
        cout << "l = " << l << ", evolution:\n";
        for (int i = 0; i < states.size(); i++) cout << "r = " << positions[i] << ", psi = " << states[i](0)<< ", d psi / dx = " << states[i](1) << "\n";
        cout << endl;

        std::ofstream file;

        file.open("../results/test_l" + std::to_string(l) + ".txt");
        for (int i = 0; i < states.size(); i++) {
            file << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << std::scientific << positions[i] << "," << std::scientific << states[i](0) << std::endl;
        }
        // don't forget to clean up :)
        file.close();
    }

}

complex<double> apw::truncated_plane_wave(Vector3d q, Vector3d r) {
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

std::vector<Vector3d> apw::generate_reciprocal_lattice(std::vector<Vector3d> lattice) {
    Vector3d & A = lattice[0];
    Vector3d & B = lattice[1];
    Vector3d & C = lattice[2];
    double V = A.transpose() * (B.cross(C)); 

    Vector3d B1 = 2*M_PI / V * B.cross(C);
    Vector3d B2 = 2*M_PI / V * C.cross(A);
    Vector3d B3 = 2*M_PI / V * A.cross(B);

    return std::vector<Vector3d>({B1, B2, B3});
}
