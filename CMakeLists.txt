cmake_minimum_required(VERSION 3.18...3.27)
project(Mesh_optimization)

set(DEV_MODULE Development.Module)

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

find_package(Eigen3 3.3.0 QUIET REQUIRED)

find_package(OpenMP COMPONENTS CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

nanobind_add_module(cgal_mesh_optimization src/cgal_mesh_optimization_interface.cpp)


target_include_directories(cgal_mesh_optimization PUBLIC cgal/Mesh_optimization/include/)
target_link_libraries(cgal_mesh_optimization PRIVATE Eigen3::Eigen)


if( OpenMP_CXX_FOUND )
    target_link_libraries(cgal_mesh_optimization PRIVATE OpenMP::OpenMP_CXX)
endif()