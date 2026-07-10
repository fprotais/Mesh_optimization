cmake_minimum_required(VERSION 3.12...3.31)
project(versatile_smoother_supplemental)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
if (MSVC)
    set(CMAKE_CXX_FLAGS "/EHsc")
    add_compile_definitions(NOMINMAX)
else()
    set(CMAKE_CXX_FLAGS "-O3 -g")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Eigen3 3.3.0 QUIET REQUIRED)

find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  if(MSVC)
    # Override the compile options for the imported OpenMP target
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
                          INTERFACE_COMPILE_OPTIONS "/openmp:llvm")
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)

# Export a header-only interface target so FetchContent users can link to it.
add_library(Mesh_optimization INTERFACE)
target_include_directories(Mesh_optimization
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Make examples optional so this project is FetchContent-friendly.
# If this CMakeLists is the top-level project, enable examples by default.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  option(BUILD_EXAMPLES "Build examples" ON)
else()
  option(BUILD_EXAMPLES "Build examples" OFF)
endif()

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
else()
  message(STATUS "Examples are disabled (BUILD_EXAMPLES=OFF).")
endif()

option(BUILD_SANDBOX "Build sandbox" OFF)
if (BUILD_SANDBOX)
    add_subdirectory(sandbox)
endif()
