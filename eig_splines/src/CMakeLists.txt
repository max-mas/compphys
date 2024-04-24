cmake_minimum_required(VERSION 3.16)
project(eig_splines)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
#find_package(Boost REQUIRED COMPONENTS math)

add_executable(eig_splines
        "./c/main.cpp"
        c/b_splines.h
        c/spherical_seq.h
        c/potentials.h
)

target_link_libraries(eig_splines PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
