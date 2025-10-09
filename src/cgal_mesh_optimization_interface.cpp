#include "CGAL/Mesh_optimization/default_structures.h"
#include "CGAL/Mesh_optimization/Mesh_conformal_optimizer.h"

#include <string>
#include <Eigen/Eigen>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>




namespace pycgal {
    static Eigen::MatrixXd const empty_points(0, 3);
    static Eigen::MatrixXi const empty_array(0, 0);
    static Eigen::VectorXi const empty_vector(0, 0);

    void validate_argument(bool condition, std::string const &message = "Assertion failed") {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

    Eigen::VectorXd optimize_mesh(nanobind::DRef<Eigen::MatrixXd const> const& points,
                                  nanobind::DRef<Eigen::VectorXi const> const& locked_indices,
                                  nanobind::DRef<Eigen::MatrixXi const> const& tetrahedra,
                                  nanobind::DRef<Eigen::MatrixXi const> const& hexahedra,
                                  nanobind::DRef<Eigen::MatrixXi const> const& pyramids,
                                  nanobind::DRef<Eigen::MatrixXi const> const& wedges,
                                  nanobind::DRef<Eigen::MatrixXd const> const& bnd_points,
                                  nanobind::DRef<Eigen::MatrixXi const> const& bnd_triangles,
                                  unsigned max_number_of_iterations = 100)
    {
        validate_argument(points.cols() == 3 || points.cols() == 2, "Points should have 2 or 3 coordinates.");
        validate_argument(tetrahedra.cols() == 4 || tetrahedra.size() == 0, "Tetrahedra should have 4 vertices.");
        validate_argument(hexahedra.cols() == 8 || hexahedra.size() == 0, "Hexahedra should have 8 vertices.");
        validate_argument(pyramids.cols() == 5 || pyramids.size() == 0, "Pyramids should have 5 vertices.");
        validate_argument(wedges.cols() == 6 || wedges.size() == 0, "Wedges should have 6 vertices.");
        validate_argument(bnd_points.cols() == 3 || bnd_points.size() == 0, "Boundary points should have 3 columns.");
        validate_argument(bnd_triangles.cols() == 3 || bnd_triangles.size() == 0, "Boundary triangles should have 3 columns.");


        return points;
    }
}

// Bind the wrapper to the Python module
void bind_optimize_mesh(nanobind::module_ &m)
{
  m.def(
    "optimize_mesh",
    &pycgal::optimize_mesh,
    "V"_a,
    "F"_a,
    "b"_a,
    "bc"_a,
    R"(Compute a Least-squares conformal map parametrization.

    @param[in] V  #V by 3 list of mesh vertex positions
    @param[in] F  #F by 3 list of mesh faces (must be triangles)
    @param[in] b  #b list of boundary indices into V
    @param[in] bc #b by 2 list of boundary values
    @param[out] V_uv #V by 2 list of 2D mesh vertex positions in UV space
    @param[out] Q  #Vx2 by #Vx2 symmetric positive semi-definite matrix for computing LSCM energy
    @return Tuple containing:
      - V_uv: UV coordinates of vertices
      - Q: Symmetric positive semi-definite matrix for LSCM energy)"
  );
}
