

create_single_source_cgal_program("cgal_offset_computation.cpp")
create_single_source_cgal_program("cgal_c3t3_test.cpp")
target_link_libraries(cgal_offset_computation PRIVATE Eigen3::Eigen)
target_link_libraries(cgal_c3t3_test PRIVATE Eigen3::Eigen)

if( OpenMP_CXX_FOUND )
    target_link_libraries(cgal_offset_computation PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(cgal_c3t3_test PRIVATE OpenMP::OpenMP_CXX)
endif()



