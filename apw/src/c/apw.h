/**
 * @file apw.h
 * @author Max Maschke (m.maschke@tu-bs.de)
 * @brief 
 * @date 2024-05-18
 * 
 * @copyright Copyright (c) 2024 Max Maschke
 * 
 */

#ifndef APW_APW_H
#define APW_APW_H

#include <boost/math/special_functions/spherical_harmonic.hpp>
using boost::math::spherical_harmonic;
#include <boost/math/special_functions/bessel.hpp> 
using boost::math::sph_bessel;
#include <boost/math/special_functions/legendre.hpp>
using boost::math::legendre_p;

#include <boost/numeric/odeint.hpp>

#include <boost/math/interpolators/cubic_b_spline.hpp>
using boost::math::cubic_b_spline;

#include <Eigen/Dense>
using Eigen::Vector3d, Eigen::Vector2d, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXcd;

#include <iostream>
using std::cout, std::endl;

#include <complex>
using std::complex, std::conj;

#include <cmath>

#include <vector>

#include <functional>

#include <fstream>

#include <iomanip>

#include <algorithm>

#include <tuple>

#include <cfenv>

#include <string>

#include "atom.h"

#include "progressbar.hpp"

class apw {

public:
    unsigned int lmax;
    unsigned int Z;
    double R;

    apw() = default;

    // for now: only 1+ ions where 1 electron removed
    apw(unsigned int chargeNum, unsigned int lMax, double muffin_tin_radius, std::vector<Vector3d> latticeVecs, double latticeConst);

    void save_muffin_tin_orbital(int l, std::string path,
                        std::vector<std::pair<std::function<double (double)>, std::function<double (double)>>> muffin_tin_functions);

    void save_dets_on_k_path(int k_steps, int E_steps, Vector3d k0, Vector3d k_diff, std::vector<Vector3d> Ks, std::string path);    

private:
    double lattice_constant;
    std::vector<Vector3d> lattice_vecs;
    std::vector<Vector3d> reciprocal_lattice_vecs;

    atom ion;

    std::vector<std::pair<std::function<double (double)>, std::function<double (double)>>> 
        get_muffin_tin_functions_E(double E);

    complex<double> psi_apw(Vector3d q, Vector3d r, cubic_b_spline<double> muffintin);

    complex<double> truncated_plane_wave(Vector3d q, Vector3d r);

    /**
     * @brief 
     * 
     * @param x 
     * @return Vector3d (r, theta, phi)
     */
    Vector3d cartesian_to_spherical(Vector3d x);

    std::vector<Vector3d> generate_reciprocal_lattice_basis(std::vector<Vector3d> lattice);

    std::vector<Vector3d> generate_finite_reciprocal_lattice_set(double norm_cutoff, int index_cutoff);

    void generate_H(MatrixXd & H, double E, Vector3d k, std::vector<Vector3d> finite_reciprocal_lattice_set,
                        std::vector<std::pair<std::function<double (double)>, std::function<double (double)>>>  muffin_tin_functions);

};


#endif //APW_APW_H