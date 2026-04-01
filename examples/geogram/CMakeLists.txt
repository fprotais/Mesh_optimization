set(GEOGRAM_LIB_ONLY ON)

FetchContent_Declare(
    geogram
    GIT_REPOSITORY https://github.com/BrunoLevy/geogram
    GIT_TAG        72bfd854d4fb6bac360689a87fc7b10db87bb5f8 #master
)
FetchContent_MakeAvailable(geogram)


add_executable(geogram_smooth geogram_smooth.cpp)
target_link_libraries(geogram_smooth PRIVATE Eigen3::Eigen geogram)
if( OpenMP_CXX_FOUND )
    target_link_libraries(geogram_smooth PRIVATE OpenMP::OpenMP_CXX)
endif()



