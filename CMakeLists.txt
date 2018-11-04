cmake_minimum_required(VERSION 3.9)
project(deep_learning)

set(CMAKE_CXX_STANDARD 11)
include_directories(/usr/local/include/eigen3)
add_executable(deep_learning main.cpp src/Matrix.h src/Layer/DenseLayer.h src/ActiveFunc/ActiveFun.h src/Loss/Loss.h src/Optimization/Optimization.h data/preprocess.h)