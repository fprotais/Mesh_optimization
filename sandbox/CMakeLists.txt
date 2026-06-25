include(FetchContent)
FetchContent_Declare(
    ultimaille
    GIT_REPOSITORY https://github.com/ultimaille/ultimaille
    GIT_TAG        master
)
FetchContent_MakeAvailable(ultimaille)
target_include_directories(ultimaille INTERFACE ${ultimaille_SOURCE_DIR}/)

add_executable(ultimaille_smooth ultimaille_smooth.cpp)
add_executable(ultimaille_surface_smooth ultimaille_surface_smooth.cpp)

target_link_libraries(ultimaille_smooth PRIVATE Eigen3::Eigen ultimaille)
target_link_libraries(ultimaille_surface_smooth PRIVATE Eigen3::Eigen ultimaille)
if( OpenMP_CXX_FOUND )
    target_link_libraries(ultimaille_smooth PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(ultimaille_surface_smooth PRIVATE OpenMP::OpenMP_CXX)
endif()