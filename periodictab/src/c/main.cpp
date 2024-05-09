//
// Created by max on 5/7/24.
//
#include <iostream>

#include "atom.h"

int main() {

    {
        atom<double> He(2, 2, 3, 100, 30, 30);
        He.run();
    }

    {
        atom<double> He_ion(2, 1, 3, 100, 30, 30);
        He_ion.run();
    }
    /*
    {
        atom<double> Ne(10, 10, 15, 100, 200, 1000);
        Ne.run();
    }

    {
        atom<double> Ne_ion(10, 9, 15, 100, 200, 1000);
        Ne_ion.run();
    }*/

    return 0;
}