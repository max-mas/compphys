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

#include <boost/numeric/odeint.hpp>

#include <Eigen/Dense>
using Eigen::Vector3d, Eigen::Vector2d, Eigen::VectorXd, Eigen::MatrixXd;

#include <iostream>
using std::cout, std::endl;

#include <complex>
using std::complex, std::conj;

#include <cmath>

#include <vector>

#include <functional>

#include <fstream>

#include <iomanip>

#include "atom.h"

class apw {

public:
    unsigned int lmax;
    unsigned int Z;
    double R;

    apw() = default;

    // for now: only 1+ ions where 1 electron removed
    apw(unsigned int chargeNum, unsigned int lMax, double muffin_tin_radius, std::vector<Vector3d> latticeVecs, double latticeConst);

private:
    double lattice_constant;
    std::vector<Vector3d> lattice_vecs;
    std::vector<Vector3d> reciprocal_lattice_vecs;
    std::vector<std::function<double (double)>> muffin_tin_solutions;

    atom ion;

    complex<double> truncated_plane_wave(Vector3d q, Vector3d r);

    /**
     * @brief 
     * 
     * @param x 
     * @return Vector3d (r, theta, phi)
     */
    Vector3d cartesian_to_spherical(Vector3d x);

    std::vector<Vector3d> generate_reciprocal_lattice(std::vector<Vector3d> lattice);



};


#endif //APW_APW_H