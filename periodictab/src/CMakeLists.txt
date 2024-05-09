cmake_minimum_required(VERSION 3.16)
project(periodictab)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
#find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)

add_executable(periodictab
        "./c/main.cpp"
        c/b_splines.h
        c/spherical_seq.h
        c/collocation.h
        c/atom.h
)

target_link_libraries(periodictab PUBLIC Eigen3::Eigen Boost::boost)
