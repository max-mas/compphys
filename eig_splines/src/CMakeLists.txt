cmake_minimum_required(VERSION 3.16)
project(poisson)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
#find_package(Boost REQUIRED COMPONENTS math)

add_executable(poisson
        "./c/main.cpp"
        c/b_splines.h
        c/collocation.h
        c/poisson_funcs.h
)

target_link_libraries(poisson PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
