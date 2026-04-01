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

add_subdirectory(examples)

