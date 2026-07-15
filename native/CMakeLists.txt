cmake_minimum_required(VERSION 3.20)

project(disparity_deblur_native LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(
  _native
  MODULE
  src/bindings.cpp
  src/tv_l1.cpp
)

target_compile_features(_native PRIVATE cxx_std_17)
target_include_directories(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(_native PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(_native PRIVATE /O2 /W4)
else()
  target_compile_options(_native PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _native LIBRARY DESTINATION disparity_deblur)
