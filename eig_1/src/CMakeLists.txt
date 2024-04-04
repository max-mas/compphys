cmake_minimum_required(VERSION 3.28)
project(eig1)

set(CMAKE_CXX_STANDARD 17)

include_directories(~/eigen/eigen/)

add_executable(eig1
        "./c/main.cpp")
