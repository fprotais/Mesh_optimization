#include <cstdlib>
#include <random>
#include <set>
#include <map>

#include <Mesh_optimization/Mesh_conformal_optimizer.h>

#include "include/ultimaille_interfaces.h"
#include "include/ultimaille_curve_query.h"
#include "include/ultimaille_boundary_query.h"
#include "include/ultimaille_mesh_utils.h"
#include <ultimaille/all.h>



int main(int argc, char** argv) {
    const std::string filename = (argc > 1) ? argv[1] : "../data/fandisk_hexalab_kenshi.vtk";
    const std::string boundaryname = (argc > 2) ? argv[2] : std::string();
    const std::string curvesname = (argc > 3) ? argv[3] : std::string();

    UM_extension::Mixed_element_mesh_serializer mixed_mesh;
    mixed_mesh.load(filename);

    UM::Triangles boundary_tri_mesh;

    if (boundaryname.empty()) {
        UM_extension::copy_surface(mixed_mesh.get_triangulated_surface(), boundary_tri_mesh);
    }
    else {
        UM::read_by_extension(boundaryname, boundary_tri_mesh);
    }

    UM::PolyLine target_curves;
    UM::PolyLine boundary_curves;
    UM::read_by_extension(filename, boundary_curves);
    if (curvesname.empty()) {
        UM::read_by_extension(filename, target_curves);
    }
    else {
        UM::read_by_extension(curvesname, target_curves);
    }

    UM_extension::Surface_projector projector(boundary_tri_mesh);

    UM::write_by_extension("boundary.mesh", boundary_tri_mesh);

    UM_extension::PolyLine_wrapper curve_wrapper{boundary_curves};
    UM_extension::Target_edge_network curve_projector(target_curves);

    mixed_mesh.save("input", ".mesh");

    Mesh_optimization::Mesh_conformal_optimizer optimizer(mixed_mesh.get_mixed_mesh(true, false, false, false), mixed_mesh.get_polygonal_boundary(), curve_wrapper);

    optimizer.set_boundary_query(projector.get_callable_custom_polygon_query());
    optimizer.set_curves_query([&](UM::vec3 const &pt, unsigned curve_id, double) -> std::tuple<UM::vec3, UM::vec3, double> {
        auto [p, n] = curve_projector.proj(pt, curve_id);
        return {p, n, 1.};
    });

    optimizer.set_verbose();
    optimizer.set_max_number_of_iteration(100);
    optimizer.run();

    mixed_mesh.save("output", ".mesh");


    return EXIT_SUCCESS;
}
