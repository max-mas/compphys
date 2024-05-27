#include <iostream>
#include "atom.h"
#include "apw.h"

int main() {
    if (true) {
        // Lithium lattice vecs:
        double lattice_const = 6.632; // (Bohr radii) ! TODO change to 6.4912 a_0 used to be 6.632
        Vector3d A, B, C;
        A << -1.0, 1.0, 1.0;
        A *= lattice_const / 2.0;
        B << 1.0, -1.0, 1.0;
        B *= lattice_const / 2.0;
        C << 1.0, 1.0, -1.0;
        C *= lattice_const / 2.0;

        std::vector<Vector3d> lattice_vecs({A, B, C});

        //apw a(3, 6, 1.30, 2.0, lattice_vecs, lattice_const, "../results/det_4_R_130/"); // what cutoff?
        apw a(3, 6, 1.30, 2.0, lattice_vecs, lattice_const, "../results/det_DOS_2_morepts/");

    }

    if (false) {
        int lmax = 3;
        int knots_seq = 200;
        // Li: 1s^2 2s^2

        Eigen::MatrixXi occmat = Eigen::MatrixXi::Zero(lmax+1, knots_seq);
        occmat(0, 0) = 2; //1s2
        occmat(0, 1) = 0; //2s1

        double eta = 0.4;

        atom atom(3, 2, 100, 50.0, knots_seq, 400, eta, 5e-4, occmat);
        atom.run();
    }
    if (false) {
        int lmax = 3;
        int knots_seq = 200;
        // cu: 1s^2 2s^2 2p^6 3s^2 3p^6 4s^1 3d^10

        Eigen::MatrixXi occmat = Eigen::MatrixXi::Zero(lmax+1, knots_seq);
        occmat(0, 0) = 2; //1s2
        occmat(0, 1) = 2; //2s2
        occmat(0, 2) = 2; //3s2
        occmat(0, 3) = 1; //4s1

        occmat(1, 1) = 6; //2p6
        occmat(1, 2) = 6; //3p6

        occmat(2, 2) = 10;//3d10

        double eta = 0.4;

        atom atom(29, 29, 100, 100.0, knots_seq, 400, eta, 5e-4, occmat);
        atom.run();
    }
    
    return 0;
}