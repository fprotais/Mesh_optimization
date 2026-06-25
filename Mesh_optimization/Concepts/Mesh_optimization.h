#pragma once

#include <vector>
#include <array>
#include <Eigen/Eigen>
#include "Mesh_conformal_optimizer.h"
#include "mesh_representations.h"

#include <iostream>

// NOTE: The following file is a WIP, do not try to use it. 

namespace Mesh_optimization {

using cgal_types::Triangulation_3_wrapper; 
using cgal_types::Surface_mesh_wrapper; 
using cgal_types::C3T3_wrapper; 

template<
    typename Triangulation_3,
    typename SurfaceMesh = cgal_types::Surface_mesh_placeholder<Triangulation_3>
    // todo: add a segment representation here. I don't think CGAL has a standard one. 
>
class Triangulation_3_optimizer : public Mesh_conformal_optimizer <Triangulation_3_wrapper<Triangulation_3>, Surface_mesh_wrapper<SurfaceMesh>>{
public:
    using Mesh_conformal_optimizer<Triangulation_3_wrapper<Triangulation_3>, Surface_mesh_wrapper<SurfaceMesh>>;

    Triangulation_3_optimizer(Triangulation_3 &tr, SurfaceMesh const &sm = SurfaceMesh()) 
    : Mesh_conformal_optimizer(Triangulation_3_wrapper<Triangulation_3>{tr}, Surface_mesh_wrapper<SurfaceMesh>{sm})
    {}
};


template<
    typename C3T3
>
class C3T3_optimizer : public Mesh_conformal_optimizer <C3T3_wrapper<C3T3>, C3T3_wrapper<C3T3>, C3T3_wrapper<C3T3>> {
public:
    using Mesh_conformal_optimizer<<C3T3_wrapper<C3T3>, C3T3_wrapper<C3T3>, C3T3_wrapper<C3T3>>;

    Triangulation_3_optimizer(C3T3 &c3t3) 
    : Mesh_conformal_optimizer(C3T3_wrapper<C3T3>{c3t3}, C3T3_wrapper<C3T3>{c3t3}, C3T3_wrapper<C3T3>{c3t3})
    {}

};

}

