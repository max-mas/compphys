cmake_minimum_required(VERSION 3.16)
project(qmc)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
#find_package(Boost REQUIRED COMPONENTS math)

add_executable(qmc
        "./c/main.cpp"
        c/monte_carlo.cpp
        c/monte_carlo.h
        c/trial_functions.cpp
        c/trial_functions.h
)

target_link_libraries(qmc PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
