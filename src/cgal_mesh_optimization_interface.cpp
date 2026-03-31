#include <Mesh_optimization/Mesh_conformal_optimizer.h>


#include <string>
#include <Eigen/Eigen>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>


using Mesh_optimization::utils::Contiguous_unsigned_range;

using Py_query = std::function<void (Eigen::MatrixXd const &coords, Eigen::VectorXi const &patch_ids, Eigen::VectorXd const &radius, Eigen::MatrixXd &results)>;

void validate_argument (bool condition, std::string const &message = "Assertion failed") {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}




struct Mesh_optimization_data {

    // input volume
    Eigen::MatrixXd points;
    Eigen::MatrixXd ref_points;
    Eigen::VectorXi locked_indices;

    Eigen::MatrixXi tetrahedra;
    Eigen::MatrixXi hexahedra;
    Eigen::MatrixXi pyramids;
    Eigen::MatrixXi wedges;
    

    // input selected elements
    Eigen::MatrixXi selected_triangles;
    Eigen::VectorXi selected_triangle_patch_ids;
    std::vector<std::vector<int>> selected_polygons;
    std::vector<unsigned> selected_polygons_patch_ids;

    Eigen::MatrixXi selected_edges;
    Eigen::VectorXi selected_edges_patch_ids;

    std::unordered_map<int, Eigen::Vector3d> point_targets;


    enum PROJECTION_SETTING {
        NONE,
        LOCKED_SELECTION, 
        QUERY_FUNCTION,
        INPUT_DATA
    };


    // projection tools
    PROJECTION_SETTING surface_projection_setting = NONE;

    Py_query surface_query_function = nullptr;
    struct Target_surface {
        Eigen::MatrixXd points;
        Eigen::MatrixXi triangles;
        Eigen::VectorXi patch_ids;
    } target_surface;


    PROJECTION_SETTING curves_projection_setting = NONE;
    Py_query curve_query_function = nullptr;
    struct Target_curve_network {
        Eigen::MatrixXd points;
        Eigen::MatrixXi edges;
        Eigen::VectorXi patch_ids;
    } target_curve_network;


    // input parameters
    unsigned max_iteration = 1000;

    double boundary_weight = 1.0;
    double minimum_edge_size = 1e-6;

    bool verbose = false;


    // outputs
    unsigned nb_bfgs_iterations = 0;
    bool result_valid = false;



    void validate() const {
        validate_argument(points.cols() == 3, "Points should have 3 coordinates.");
        validate_argument((ref_points.cols() == 3 && ref_points.rows() == points.rows()) || ref_points.size() == 0, "Reference points should have 3 coordinates and the same number of rows as points.");

        validate_argument(tetrahedra.cols() == 4 || tetrahedra.size() == 0, "Tetrahedra should have 4 vertices.");
        validate_argument(hexahedra.cols() == 8 || hexahedra.size() == 0, "Hexahedra should have 8 vertices.");
        validate_argument(pyramids.cols() == 5 || pyramids.size() == 0, "Pyramids should have 5 vertices.");
        validate_argument(wedges.cols() == 6 || wedges.size() == 0, "Wedges should have 6 vertices.");


        validate_argument(selected_triangles.cols() == 3 || selected_triangles.size() == 0, "Selected triangles should have 3 columns.");
        validate_argument((selected_triangle_patch_ids.cols() == 3 && selected_triangle_patch_ids.rows() == selected_triangles.rows()) || selected_triangle_patch_ids.size() == 0, "Selected triangle patch ids should have the same number of rows as selected triangles.");
        validate_argument(selected_edges.cols() == 2 || selected_edges.size() == 0, "Selected edges should have 2 columns.");
        validate_argument((selected_edges_patch_ids.cols() == 2 && selected_edges_patch_ids.rows() == selected_edges.rows()) || selected_edges_patch_ids.size() == 0, "Selected edges patch ids should have the same number of rows as selected edges.");
    
        validate_argument(surface_projection_setting != QUERY_FUNCTION || surface_query_function != nullptr, "Surface projection setting is QUERY_FUNCTION but no query function is provided.");
        validate_argument(curves_projection_setting != QUERY_FUNCTION || curve_query_function != nullptr, "Curve projection setting is QUERY_FUNCTION but no query function is provided.");

        validate_argument(target_surface.points.cols() == 3 || target_surface.points.size() == 0, "Target surface points should have 3 columns.");
        validate_argument(target_surface.triangles.cols() == 3 || target_surface.triangles.size() == 0, "Target surface triangles should have 3 columns.");
        validate_argument((target_surface.patch_ids.rows() == target_surface.triangles.rows()) || target_surface.patch_ids.size() == 0, "Target surface patch ids should have the same number of rows as target surface triangles.");
        validate_argument(target_curve_network.points.cols() == 3 || target_curve_network.points.size() == 0, "Target curve network points should have 3 columns.");
        validate_argument(target_curve_network.edges.cols() == 2 || target_curve_network.edges.size() == 0, "Target curve network edges should have 2 columns.");
        validate_argument((target_curve_network.patch_ids.rows() == target_curve_network.edges.rows()) || target_curve_network.patch_ids.size() == 0, "Target curve network patch ids should have the same number of rows as target curve network edges.");

        validate_argument(boundary_weight >= 0, "Boundary weight should be non-negative.");
        validate_argument(minimum_edge_size > 0, "Minimum edge size should be positive.");

    }
};


class Mesh_volume_interface : public Mesh_optimization::helper_structures::Mixed_element_mesh<int, int, Eigen::Vector3d, Contiguous_unsigned_range> {
public:
    std::size_t nb_vertices() const override { return _coords.rows(); }

    Eigen::Vector3d vertex_coordinates(int vertex) const override {
        return _coords.row(vertex);
    }
    void set_new_vertex_coordinates(int vertex, Eigen::Vector3d coord) override { _coords.row(vertex) = coord; }

    Contiguous_unsigned_range input_cell_range() const override { return Contiguous_unsigned_range{0, _elements.size()}; }

    int get_cell_vertex(int cell, unsigned local_Vertex_descriptor) const override {
        return (*_elements[cell].mesh_storage)(_elements[cell].loc_id, local_Vertex_descriptor);
    };

    Eigen::Vector3d get_ref_vertex_coordinates(int vertex) const override {
        assert(this->has_reference_mesh);
        return _data.ref_points.row(vertex);
    }

    Shape const * get_element_shape(int cell) const override {
        return _elements[cell].shape;
    }

public:
    Mesh_volume_interface(Mesh_optimization_data const &data, Eigen::MatrixXd &coords)
    : _data(data)
    , _coords(coords)
    {
        this->has_reference_mesh = (_data.ref_points.rows() == _data.points.rows());
        for (int i = 0; i < _data.tetrahedra.rows(); ++i) {
            _elements.push_back(Mesh_element{i, &_data.tetrahedra, &tet_ref});
        }
        for (int i = 0; i < _data.hexahedra.rows(); ++i) {
            _elements.push_back(Mesh_element{i, &_data.hexahedra, &hex_ref});
        }
        for (int i = 0; i < _data.pyramids.rows(); ++i) {
            _elements.push_back(Mesh_element{i, &_data.pyramids, &  py_ref});
        }
        for (int i = 0; i < _data.wedges.rows(); ++i) {
            _elements.push_back(Mesh_element{i, &_data.wedges, & we_ref});
        }

        this->assemble(); // CRITICAL
    }

    Mesh_optimization_data const &_data;
    Eigen::MatrixXd &_coords;

    struct Mesh_element {
        int loc_id;
        Eigen::MatrixXi const *mesh_storage;
        Shape const *shape;
    };
    std::vector<Mesh_element> _elements;


    Mesh_optimization::Shapes::VTK_TETRAHEDRON<Eigen::Vector3d> tet_ref;
    Mesh_optimization::Shapes::VTK_HEXAHEDRON<Eigen::Vector3d> hex_ref;
    Mesh_optimization::Shapes::VTK_PYRAMID<Eigen::Vector3d> py_ref;
    Mesh_optimization::Shapes::VTK_WEDGE<Eigen::Vector3d> we_ref;
};



struct Mesh_surface_interface : public Mesh_optimization::helper_structures::Polygonal_boundary<int, int, Eigen::Vector3d> {
    Mesh_surface_interface(Mesh_optimization_data const &data)
    : _data(data)
    {
        for (int i = 0; i < _data.selected_triangles.rows(); ++i) {
            std::vector<int> tri = {
                _data.selected_triangles(i, 0),
                _data.selected_triangles(i, 1),
                _data.selected_triangles(i, 2)
            };
            this->add_polygon(tri, _data.selected_triangle_patch_ids.rows() == _data.selected_triangles.rows() ? _data.selected_triangle_patch_ids(i) : 0);
        }

        for (int i = 0; i < _data.selected_polygons.size(); ++i) {
            this->add_polygon(_data.selected_polygons[i], _data.selected_polygons_patch_ids.size() == _data.selected_polygons.size() ? _data.selected_polygons_patch_ids[i] : 0);
        }

        if (_data.surface_projection_setting == Mesh_optimization_data::QUERY_FUNCTION) {
            has_query = true;
            q_coords = Eigen::MatrixXd(this->nb_faces(), 3);
            q_patch_ids = Eigen::VectorXi(this->nb_faces());
            q_radius = Eigen::VectorXd(this->nb_faces());
            q_results = Eigen::MatrixXd(this->nb_faces(), 6);
            for (int i = 0; i < this->nb_faces(); ++i) {
                q_patch_ids(i) = this->surface_id(i);
            }
        }

        if (_data.surface_projection_setting == Mesh_optimization_data::INPUT_DATA) {
            // todo
        }

    }

    Eigen::MatrixXd q_coords;
    Eigen::VectorXi q_patch_ids;
    Eigen::VectorXd q_radius;
    Eigen::MatrixXd q_results;

    bool has_query = false;
    
    using Tangent = std::tuple<Eigen::Vector3d, Eigen::Vector3d>;
    
    std::function<void (std::vector<Eigen::Vector3d> const &coord, std::vector<unsigned> &surface_id, std::vector<double> &radius, std::vector<Tangent> &results)>
    get_boundary_query_function() {
        if (_data.surface_projection_setting == Mesh_optimization_data::QUERY_FUNCTION) {
            return [&](std::vector<Eigen::Vector3d> const &coord, std::vector<unsigned> &surface_id, std::vector<double> &radius, std::vector<Tangent> &results) {
                for (int i = 0; i < this->nb_faces(); ++i) {
                    for (unsigned j = 0; j < 3; ++j) {
                        q_coords(i, j) = coord[i](j);
                    }
                    q_radius(i) = radius[i];
                }
                _data.surface_query_function(q_coords, q_patch_ids, q_radius, q_results);
                for (int i = 0; i < this->nb_faces(); ++i) {
                    std::get<0>(results[i]) = { q_results(i, 0), q_results(i, 1), q_results(i, 2) };
                    std::get<1>(results[i]) = { q_results(i, 3), q_results(i, 4), q_results(i, 5) };
                }
            };
        }
        if (_data.surface_projection_setting == Mesh_optimization_data::INPUT_DATA) {
            // todo
        }
        return nullptr;
    }
    
    Mesh_optimization_data const &_data;
};

struct Mesh_edges_interface {
    using Edge_descriptor = int;
    std::size_t nb_edges() const { return _data.selected_edges.rows(); }
    Contiguous_unsigned_range edge_range() const { return Contiguous_unsigned_range{0, nb_edges()}; }
    unsigned curve_id(int edge) const { return _data.selected_edges.rows() == _data.selected_edges_patch_ids.rows() ? _data.selected_edges_patch_ids(edge) : 0; }
    int edge_vertex(int edge, unsigned i) const { return _data.selected_edges(edge, i); }


    Eigen::MatrixXd q_coords;
    Eigen::VectorXi q_curve_ids;
    Eigen::VectorXd q_radius;
    Eigen::MatrixXd q_results;

    bool has_query = false;
    Mesh_edges_interface(Mesh_optimization_data const &data)
    : _data(data)
    {
        if (_data.curves_projection_setting == Mesh_optimization_data::QUERY_FUNCTION) {
            has_query = true;
            q_coords = Eigen::MatrixXd(this->nb_edges(), 3);
            q_curve_ids = Eigen::VectorXi(this->nb_edges());
            q_radius = Eigen::VectorXd(this->nb_edges());
            q_results = Eigen::MatrixXd(this->nb_edges(), 6);
            for (int i = 0; i < this->nb_edges(); ++i) {
                q_curve_ids(i) = this->curve_id(i);
            }
        }

        if (_data.curves_projection_setting == Mesh_optimization_data::INPUT_DATA) {
            // todo
        }

    }

    using Tangent = std::tuple<Eigen::Vector3d, Eigen::Vector3d>;

    std::function<void (std::vector<Eigen::Vector3d> const &coord, std::vector<unsigned> &surface_id, std::vector<double> &radius, std::vector<Tangent> &results)>
    get_boundary_query_function() {
        if (_data.curves_projection_setting == Mesh_optimization_data::QUERY_FUNCTION) {
            return [&](std::vector<Eigen::Vector3d> const &coord, std::vector<unsigned> &surface_id, std::vector<double> &radius, std::vector<Tangent> &results) {
                for (int i = 0; i < this->nb_edges(); ++i) {
                    for (unsigned j = 0; j < 3; ++j) {
                        q_coords(i, j) = coord[i](j);
                    }
                    q_radius(i) = radius[i];
                }
                _data.curve_query_function(q_coords, q_curve_ids, q_radius, q_results);
                for (int i = 0; i < this->nb_edges(); ++i) {
                    std::get<0>(results[i]) = { q_results(i, 0), q_results(i, 1), q_results(i, 2) };
                    std::get<1>(results[i]) = { q_results(i, 3), q_results(i, 4), q_results(i, 5) };
                }
            };
        }
        if (_data.curves_projection_setting == Mesh_optimization_data::INPUT_DATA) {
            // todo
        }
        return nullptr;
    }

    Mesh_optimization_data const &_data;
};

Eigen::MatrixXd optimize_mesh(Mesh_optimization_data &data) {
    data.validate();
    Eigen::MatrixXd coords = data.points;  // Copy points
    
    Mesh_volume_interface mesh(data, coords);
    Mesh_surface_interface surface(data);
    Mesh_edges_interface curve_network(data);

    Mesh_optimization::Mesh_conformal_optimizer optimizer(mesh, surface, curve_network);

    optimizer.set_locked_vertices(data.locked_indices);
    optimizer.set_vertex_target_positions(data.point_targets);

    optimizer.set_max_number_of_iteration(data.max_iteration);
    optimizer.set_minimum_valid_edge_size(data.minimum_edge_size);
    optimizer.set_boundary_weight(data.boundary_weight);
    optimizer.set_verbose(data.verbose);

    optimizer.set_locked_boundary(data.surface_projection_setting == Mesh_optimization_data::LOCKED_SELECTION);
    if (data.curves_projection_setting == Mesh_optimization_data::LOCKED_SELECTION) {
        for (auto e : curve_network.edge_range()) {
            for (unsigned i=0; i<2; ++i) {
                optimizer.set_vertex_Lock(curve_network.edge_vertex(e, i), true);
            }
        }
    }


    if (surface.has_query) {
        optimizer.set_boundary_query(surface.get_boundary_query_function());
    }
    if (curve_network.has_query) {
        optimizer.set_boundary_query(curve_network.get_boundary_query_function());
    }


    data.result_valid = optimizer.untangle();

    return coords;
}



void bind_optimize_mesh(nanobind::module_ &m)
{
    nanobind::enum_<Mesh_optimization_data::PROJECTION_SETTING>(m, "ProjectionSetting")
        .value("NONE", Mesh_optimization_data::NONE)
        .value("LOCKED_SELECTION", Mesh_optimization_data::LOCKED_SELECTION)
        .value("QUERY_FUNCTION", Mesh_optimization_data::QUERY_FUNCTION)
        .value("INPUT_DATA", Mesh_optimization_data::INPUT_DATA);

    nanobind::class_<Mesh_optimization_data::Target_surface>(m, "TargetSurface")
        .def(nanobind::init<>())
        .def_rw("points", &Mesh_optimization_data::Target_surface::points)
        .def_rw("triangles", &Mesh_optimization_data::Target_surface::triangles)
        .def_rw("patch_ids", &Mesh_optimization_data::Target_surface::patch_ids);

    nanobind::class_<Mesh_optimization_data::Target_curve_network>(m, "TargetCurveNetwork")
        .def(nanobind::init<>())
        .def_rw("points", &Mesh_optimization_data::Target_curve_network::points)
        .def_rw("edges", &Mesh_optimization_data::Target_curve_network::edges)
        .def_rw("patch_ids", &Mesh_optimization_data::Target_curve_network::patch_ids);

    nanobind::class_<Mesh_optimization_data>(m, "MeshOptimizationData")
        .def(nanobind::init<>())
        .def_rw("points", &Mesh_optimization_data::points)
        .def_rw("ref_points", &Mesh_optimization_data::ref_points)
        .def_rw("locked_indices", &Mesh_optimization_data::locked_indices)
        .def_rw("tetrahedra", &Mesh_optimization_data::tetrahedra)
        .def_rw("hexahedra", &Mesh_optimization_data::hexahedra)
        .def_rw("pyramids", &Mesh_optimization_data::pyramids)
        .def_rw("wedges", &Mesh_optimization_data::wedges)
        .def_rw("selected_triangles", &Mesh_optimization_data::selected_triangles)
        .def_rw("selected_triangle_patch_ids", &Mesh_optimization_data::selected_triangle_patch_ids)
        .def_rw("selected_polygons", &Mesh_optimization_data::selected_polygons)
        .def_rw("selected_polygons_patch_ids", &Mesh_optimization_data::selected_polygons_patch_ids)
        .def_rw("selected_edges", &Mesh_optimization_data::selected_edges)
        .def_rw("selected_edges_patch_ids", &Mesh_optimization_data::selected_edges_patch_ids)
        .def_rw("point_targets", &Mesh_optimization_data::point_targets)
        .def_rw("surface_projection_setting", &Mesh_optimization_data::surface_projection_setting)
        .def_rw("set_surface_query", &Mesh_optimization_data::surface_query_function)
        .def_rw("target_surface", &Mesh_optimization_data::target_surface)
        .def_rw("curves_projection_setting", &Mesh_optimization_data::curves_projection_setting)
        .def_rw("set_curve_query", &Mesh_optimization_data::curve_query_function)
        .def_rw("target_curve_network", &Mesh_optimization_data::target_curve_network)
        .def_rw("max_iteration", &Mesh_optimization_data::max_iteration)
        .def_rw("boundary_weight", &Mesh_optimization_data::boundary_weight)
        .def_rw("minimum_edge_size", &Mesh_optimization_data::minimum_edge_size)
        .def_rw("verbose", &Mesh_optimization_data::verbose);

    m.def("optimize_mesh", &optimize_mesh, nanobind::arg("data"),
          R"(Optimize a mesh using conformal mapping.

          @param[in] data  MeshOptimizationData struct containing all mesh and optimization parameters
          @return Optimized vertex positions as a new matrix)");
}

NB_MODULE(cgal_mesh_optimization, m) {
    bind_optimize_mesh(m);
}



