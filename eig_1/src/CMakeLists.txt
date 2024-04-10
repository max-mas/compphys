cmake_minimum_required(VERSION 3.16)
project(eig1)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_executable(eig1
        "./c/main.cpp"
        c/discrete_hamiltonian.h
        c/discrete_hamiltonian.cpp
        c/potentials.cpp
        c/potentials.h
        c/potentials.cpp
        c/solve.cpp
        c/solve.h
        c/out.cpp
        c/out.h
        c/benchmark.cpp
        c/benchmark.h
        c/inverse_iteration.cpp
        c/inverse_iteration.h
        c/observables.cpp
        c/observables.h
)

target_link_libraries(eig1 PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
