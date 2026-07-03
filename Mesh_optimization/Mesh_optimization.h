#pragma once

#include <vector>
#include <array>
#include <Eigen/Eigen>
#include "Mesh_conformal_optimizer.h"
#include "mesh_representations.h"

#include <iostream>

// NOTE: The following file is a WIP, do not try to use it. 

namespace Mesh_optimization {

using cgal_types::C3T3_wrapper; 


template<
    typename C3T3
>
class C3T3_optimizer : public Mesh_conformal_optimizer <C3T3_wrapper<C3T3>, C3T3_wrapper<C3T3>, C3T3_wrapper<C3T3>> {
private:
    C3T3_wrapper<C3T3> mesh_wrapper;

public:
    C3T3_optimizer(C3T3 &c3t3) 
    : mesh_wrapper{c3t3}
    , Mesh_conformal_optimizer<C3T3_wrapper<C3T3>, C3T3_wrapper<C3T3>, C3T3_wrapper<C3T3>>(mesh_wrapper, mesh_wrapper, mesh_wrapper)
    {}

};

}

