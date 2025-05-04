cmake_minimum_required(VERSION 3.16)
project(qp_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/lib/eigen-3.4.0
)

add_executable(qp_main src/main.cpp src/QP.cpp)

find_package(pybind11 REQUIRED)
pybind11_add_module(qp_module src/qp_bindings.cpp src/QP.cpp)