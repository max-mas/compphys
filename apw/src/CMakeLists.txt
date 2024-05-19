cmake_minimum_required(VERSION 3.16)
project(apw)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
#find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)

add_executable(apw
        "./c/main.cpp"
        c/apw.h
        c/apw.cpp
        c/b_splines.h
        c/b_splines.cpp
        c/spherical_seq.h
        c/spherical_seq.cpp
        c/collocation.h
        c/collocation.cpp
        c/atom.h
        c/atom.cpp
)

target_link_libraries(apw PUBLIC Eigen3::Eigen Boost::boost)
