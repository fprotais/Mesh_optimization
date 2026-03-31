#pragma once
#include <cassert>
#include <unordered_map>
#include <functional>
#include <Eigen/Eigen>
#include "mesh_representations.h"
#include "internal/Tetrahedral_conformal_optimizer.h"

namespace Mesh_optimization {

template<
    typename TetrahedralMesh = default_structures::Empty_mesh,
    typename BoundaryMesh = default_structures::Empty_boundary<typename TetrahedralMesh::Vertex_descriptor>,
    typename EdgeNetwork = default_structures::Empty_edge_network<typename TetrahedralMesh::Vertex_descriptor>
>
class Mesh_conformal_optimizer {
public:
    using Vertex_descriptor = typename TetrahedralMesh::Vertex_descriptor;
    using Cell_descriptor = typename TetrahedralMesh::Cell_descriptor;
    using Point_3 = typename TetrahedralMesh::Point_3;
    using Normal_3 = typename BoundaryMesh::Normal_3;
    template<typename T> using Vertex_descriptor_map = std::unordered_map<Vertex_descriptor, T>; // todo: manage to template that?
    template<typename T> using Cell_descriptor_map = std::unordered_map<Cell_descriptor, T>;

    Mesh_conformal_optimizer(TetrahedralMesh &mesh, BoundaryMesh const &boundary = BoundaryMesh(), EdgeNetwork const &edge_network = EdgeNetwork());

    void set_locked_boundary(bool locked = true);

    void set_verbose(bool verbose = true);
    void set_max_number_of_iteration(unsigned number_of_iterations);

    // todo: locks are relatively inefficient because they use a map. Should it be improved?
    void set_vertex_Lock(Vertex_descriptor vertex, bool locked = true);
    void set_vertex_dim_lock(Vertex_descriptor vertex, unsigned dimension, bool locked = true);
    void set_vertices_dim_locks(Vertex_descriptor_map<std::array<bool, 3>> const &vertex_dimension_locks);

    template <typename Container>
    void set_locked_vertices(Container const &vertices) {
        for (Vertex_descriptor vertex : vertices) {
            set_vertex_Lock(vertex, true);
        }
    }

    void clear_locks();

    // QUERIES FOR BOUNDARY PROJECTION
    using Plane = std::tuple<Point_3, Normal_3>;

    using Boundary_point_query = std::function<Plane (Point_3 const &coord, unsigned surface_id, double radius)>;
    using Boundary_polygon_query = std::function<Plane (std::vector<Point_3> const &triangle, unsigned surface_id)>;
    using Boundary_point_batch_query = std::function<void (std::vector<Point_3> const &coord, std::vector<unsigned> &surface_id, std::vector<double> &radius, std::vector<Plane> &results)>;
    using Boundary_polygon_batch_query = std::function<void (std::vector<std::vector<Point_3>> const &triangles, std::vector<unsigned> &surface_id, std::vector<Plane> &results)>;

    // use the last setting that was called
    // batch will call every point at every iteration, while singular will call only the needed ones.
    // Important:
    //   - singular calls must be thread safe.
    //   - batch will not change the behavior: only needed values will be used (i.e. use only if your singular calls are particularly slow).
    //   - point_query will query at the center of the polygon with its area as a guess of the radius. Favor the polygon query when possible.
    void set_boundary_query(Boundary_point_query boundary_query);
    void set_boundary_query(Boundary_polygon_query boundary_query);
    void set_boundary_query(Boundary_point_batch_query boundary_query);
    void set_boundary_query(Boundary_polygon_batch_query boundary_query);


    // QUERIES FOR CURVE NETWORK PROJECTION
    using Curve_tangent = std::tuple<Point_3, Normal_3>;

    using Curve_point_query = std::function<Curve_tangent (Point_3 const &coord, unsigned curve_id, double radius)>;
    using Curve_segment_query = std::function<Curve_tangent (std::array<Point_3, 2> const &edge, unsigned curve_id)>;
    using Curve_point_batch_query = std::function<void (std::vector<Point_3> const &coord, std::vector<unsigned> &curve_ids, std::vector<double> &radius, std::vector<Curve_tangent> &results)>;
    using Curve_segment_batch_query = std::function<void (std::vector<std::array<Point_3, 2>> const &edges, std::vector<unsigned> &curve_ids, std::vector<Curve_tangent> &results)>;

    // see set_boundary_query for comments on how to use those functions
    // note that for simplicity purposes, curve query and boundary term are purely serial. If you have a use case where it is limiting, it is something we can change
    void set_curves_query(Curve_point_query boundary_query);
    void set_curves_query(Curve_segment_query boundary_query);
    void set_curves_query(Curve_point_batch_query boundary_query);
    void set_curves_query(Curve_segment_batch_query boundary_query);

    // ADDING QUADRATIC TARGETS FOR VERTICES

    // quadratic energy minimization towards target positions
    void set_vertex_target_position(Vertex_descriptor v, Point_3 const &target_positions);
    void set_vertex_target_positions(std::vector<std::pair<Vertex_descriptor, Point_3>> const &target_positions);

    template <typename Container>
    void set_vertex_target_positions(Container const &target_positions) {
        for (auto const & [v, pt] : target_positions) {
           set_vertex_target_position(v, pt);
        }
    }

    void clear_vertex_target_positions();

    void naive_smooth(); // runs gradient based laplacian smoothing

    bool untangle();

    bool maximize_quality(); // currently not well tested with boundary constraints

public: // for advanced usage. Do not touch if you do not know what you are doing.

    void set_minimum_valid_edge_size(double edge_size);  // should be a minimum bound on the valid edge size of the mesh, used as a reference for untangling

    void set_boundary_weight(double weight); // large values can lead to convergence issues

    unsigned get_total_number_of_lbfgs_iterations() const;

public: // for advanced monitoring

    using Iteration_status = Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Iteration_status;
    using Vertex_status = Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Vertex_status;
    using Cell_status = Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Tetrahedron_status;

    using Callback_function = std::function<bool (Iteration_status const &status,
                                                  Vertex_descriptor_map<Vertex_status> const &vertex_data,
                                                  Cell_descriptor_map<Cell_status> const &cell_data
                                                 )>;
    using Callback_setting = Mesh_optimization_internal::Tetrahedral_conformal_optimizer::DEBUG_CALLBACK_SETTING;
    void set_callback_function(Callback_function callback_function, Callback_setting setting = Callback_setting::OUTER_ITER);


private:
    // inputs
    TetrahedralMesh &_mesh;
    Vertex_descriptor_map<std::array<bool, 3>> _user_locks;
    BoundaryMesh const &_boundary;
    EdgeNetwork const &_edge_network;
    std::vector<std::pair<Vertex_descriptor, Point_3>> _vertex_target_positions;

    // options
    bool _verbose = false;
    bool _lock_boundary = true;

    unsigned _max_number_of_iteration = 1000;
    double _min_valid_edge_size = 1e-6;
    double _boundary_weight = 1.;

    bool is_vert_locked(Vertex_descriptor v) const;

    template<typename T>
    Eigen::Vector3d convert_to_eigen(T const &vector) const { return Eigen::Vector3d(vector[0], vector[1], vector[2]); }

    inline Point_3 convert_to_user(Eigen::Vector3d point) const { point = _scale*point + _shift; return Point_3(point[0], point[1], point[2]); }

    inline Eigen::Vector3d convert_to_inner(Point_3 const &point) const { return (convert_to_eigen(point) - _shift) / _scale; }


    // internal working data
    void check_refs();
    void clear_internal_data();
    void initialize_boundary();
    void initialize_curve_network();
    void create_compress_sorted_data();
    void initialise_point_targets();
    Eigen::VectorXd _compressed_coords;
    std::vector<bool> _compressed_locks;
    Vertex_descriptor_map<unsigned> _vertex_original_to_compressed;
    Cell_descriptor_map<unsigned> _cell_original_to_compressed;
    std::vector<std::array<unsigned, 4>> _tetrahedra;
    std::vector<std::array<Eigen::Vector3d, 4>> _tetrahedron_refs;
    std::vector<std::vector<unsigned>> _vert2tet_corner;


    enum QUERY_TYPE {NONE, POINT_QUERY, ELEMENT_QUERY, BATCH_POINT_QUERY, BATCH_ELEMENT_QUERY};

    std::vector<std::vector<unsigned>> _bnd_faces;
    std::vector<std::pair<unsigned, std::vector<std::array<unsigned, 2>>>> _vert_and_face_corners;
    std::vector<unsigned> _face_surface_id;
    QUERY_TYPE _boundary_query_type = NONE;
    Boundary_point_query _boundary_point_query = nullptr;
    Boundary_polygon_query _boundary_polygon_query = nullptr;
    Boundary_point_batch_query _boundary_point_batch_query = nullptr;
    Boundary_polygon_batch_query _boundary_polygon_batch_query = nullptr;
    std::vector<Point_3> _boundary_batch_info_points;
    std::vector<double> _boundary_batch_info_radii;
    std::vector<std::vector<Point_3>> _boundary_batch_info_polygons;
    std::vector<Plane> _boundary_batch_planes;

    std::vector<std::array<unsigned, 2>> _curve_edges;
    std::vector<unsigned> _curve_ids;
    QUERY_TYPE _curve_query_type = NONE;
    Curve_point_query _curve_point_query = nullptr;
    Curve_segment_query _curve_segment_query = nullptr;
    Curve_point_batch_query _curve_point_batch_query = nullptr;
    Curve_segment_batch_query _curve_segment_batch_query = nullptr;
    std::vector<Point_3> _curve_batch_info_points;
    std::vector<double> _curve_batch_info_radii;
    std::vector<std::array<Point_3, 2>> _curve_batch_info_edges;
    std::vector<Curve_tangent> _curve_batch_tangents;

    std::vector<std::pair<unsigned, Eigen::Vector3d>> _point_targets;

    double _scale = 1.;
    Eigen::Vector3d _shift = Eigen::Vector3d::Zero();
    double const _rescaled_bbox_scale = 0.5;
    void rescale_geometry();
    void update_mesh_coordinates();


    Callback_function _callback_function = nullptr;
    Callback_setting _callback_setting = Callback_setting::NOTHING;
    bool _callback_initialized = false;
    Iteration_status _callback_status;
    Vertex_descriptor_map<Vertex_status> _callback_vertex_map_data;
    Cell_descriptor_map<Cell_status> _callback_cell_map_data;
    void initialize_callback();
    bool run_callback(Iteration_status const &, std::vector<Vertex_status> const &, std::vector<Cell_status> const&);

    unsigned _nb_lbfgs_iterations = 0;

    void initialise_optimizer(Mesh_optimization_internal::Tetrahedral_conformal_optimizer &);
    void initialise_boundary_query(Mesh_optimization_internal::Tetrahedral_conformal_optimizer &);
    void initialise_curve_queries(Mesh_optimization_internal::Tetrahedral_conformal_optimizer &);
};

}

#include "internal/Mesh_conformal_optimizer_impl.hpp"
