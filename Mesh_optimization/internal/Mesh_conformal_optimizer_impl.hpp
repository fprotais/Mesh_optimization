
#include "math_functions.h"
#include "utils/colorized_text.h"

namespace Mesh_optimization {

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::Mesh_conformal_optimizer(TetrahedralMesh &mesh, BoundaryMesh const &boundary, EdgeNetwork const &edge_network)
: _mesh(mesh)
, _boundary(boundary)
, _edge_network(edge_network)
{}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_locked_boundary(bool locked) {
    _lock_boundary = locked;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_verbose(bool verbose) {
    _verbose = verbose;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_max_number_of_iteration(unsigned number_of_iterations) {
    _max_number_of_iteration = number_of_iterations;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_minimum_valid_edge_size(double val) {
    _min_valid_edge_size = val;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_boundary_weight(double val) {
    _boundary_weight = val;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
unsigned Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::get_total_number_of_lbfgs_iterations() const {
    return _nb_lbfgs_iterations;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_vertex_Lock(Vertex_descriptor vertex_descriptor, bool locked) {
    auto res = _user_locks.emplace(vertex_descriptor, std::array<bool, 3>{false, false, false});
    (*res.first).second = {locked, locked, locked};
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_vertex_dim_lock(Vertex_descriptor vertex_descriptor, unsigned dimension, bool locked) {
    assert(dimension < 3);
    auto res = _user_locks.emplace(vertex_descriptor, std::array<bool, 3>{false, false, false});
    (*res.first).second[dimension] = locked;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_vertices_dim_locks(Vertex_descriptor_map<std::array<bool, 3>> const &vertex_dimension_locks)  {
    _user_locks = vertex_dimension_locks;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::clear_locks()  {
    _user_locks.clear();
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_boundary_query(Boundary_point_query boundary_query) {
    _lock_boundary = false;
    _boundary_query_type = POINT_QUERY;
    _boundary_point_query = boundary_query;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_boundary_query(Boundary_polygon_query boundary_query) {
    _lock_boundary = false;
    _boundary_query_type = ELEMENT_QUERY;
    _boundary_polygon_query = boundary_query;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_boundary_query(Boundary_point_batch_query boundary_query) {
    _lock_boundary = false;
    _boundary_query_type = BATCH_POINT_QUERY;
    _boundary_point_batch_query = boundary_query;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_boundary_query(Boundary_polygon_batch_query boundary_query) {
    _lock_boundary = false;
    _boundary_query_type = BATCH_ELEMENT_QUERY;
    _boundary_polygon_batch_query = boundary_query;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_curves_query(Curve_point_query curve_query) {
    _lock_boundary = false;
    _curve_query_type = POINT_QUERY;
    _curve_point_query = curve_query;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_curves_query(Curve_segment_query curve_query) {
    _lock_boundary = false;
    _curve_query_type = ELEMENT_QUERY;
    _curve_segment_query = curve_query;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_curves_query(Curve_point_batch_query curve_query) {
    _lock_boundary = false;
    _curve_query_type = BATCH_POINT_QUERY;
    _curve_point_batch_query = curve_query;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_curves_query(Curve_segment_batch_query curve_query) {
    _lock_boundary = false;
    _curve_query_type = BATCH_ELEMENT_QUERY;
    _curve_segment_batch_query = curve_query;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_vertex_target_position(Vertex_descriptor v, Point_3 const &target_position) {
    _vertex_target_positions.push_back({v, target_position});
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_vertex_target_positions(std::vector<std::pair<Vertex_descriptor, Point_3>> const &target_positions) {
    _vertex_target_positions = target_positions;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::clear_vertex_target_positions() {
    _vertex_target_positions.clear();
}


template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
bool Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::is_vert_locked(Vertex_descriptor Vertex_descriptor) const {
    auto iter = _user_locks.find(Vertex_descriptor);
    if (iter == _user_locks.end()) return false;
    return (*iter).second[0] && (*iter).second[1] && (*iter).second[2];
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::check_refs() {
    // when is it that the input tetrahedra is wrong, or is it just tangled?
    // bool input_contains_invalid_references = false;
    // for (auto cell : _mesh.cell_range()) {
    //     auto ref = _mesh.cell_reference_shape(cell);
    //     std::array<Eigen::Vector3d, 4> tet;
    //     for (unsigned i = 0; i < 4; ++i) {
    //         tet[i] = convert_to_eigen(ref[i]);
    //     }
    //     if ((tet[1]-tet[0]).cross(tet[2]-tet[0]).dot(tet[3]-tet[0]) <= 0) {
    //         input_contains_invalid_references = true;
    //     }
    // }
    // assert(!input_contains_invalid_references);
    // if (input_contains_invalid_references && _verbose) {
    //     Mesh_optimization_internal::Colorized_print("Mesh_conformal_optimizer: some tetrahedron reference shapes are invalid (negative volume). Results may be incorrect.", Mesh_optimization_internal::ConsoleTextColor::Red);
    // }
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::clear_internal_data() {
    _compressed_coords.resize(0);
    _compressed_locks.clear();
    _vertex_original_to_compressed.clear();
    _cell_original_to_compressed.clear();
    _tetrahedra.clear();
    _tetrahedron_refs.clear();
    _vert2tet_corner.clear();

    _bnd_faces.clear();
    _vert_and_face_corners.clear();
    _face_surface_id.clear();

    _curve_edges.clear();
    _curve_ids.clear();
    _point_targets.clear();

    _callback_initialized = false;
    _callback_vertex_map_data.clear();
    _callback_cell_map_data.clear();

    _nb_lbfgs_iterations = 0;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::create_compress_sorted_data() {
    clear_internal_data();
    if (_verbose) std::cout << "Mesh_conformal_optimizer: copying and compressing data " << std::endl;
    std::size_t nb_points = 0;
    _vertex_original_to_compressed.reserve(_mesh.nb_vertices());
    std::vector<unsigned> nb_tet_on_verts;
    std::vector<double> temp_coordinates_storage;
    nb_tet_on_verts.reserve(_mesh.nb_vertices());
    _compressed_locks.reserve(3*_mesh.nb_vertices());
    temp_coordinates_storage.reserve(3*_mesh.nb_vertices());

    auto get_compressed_point_id = [&](Vertex_descriptor Vertex_descriptor) {
        auto res = _vertex_original_to_compressed.emplace(Vertex_descriptor, nb_points);
        if (res.second) {
            ++nb_points;
            nb_tet_on_verts.push_back(0);
            Point_3 coordinates = _mesh.vertex_coordinates(Vertex_descriptor);
            auto iterator = _user_locks.find(Vertex_descriptor);
            for (unsigned i = 0; i < 3; ++i) {
                temp_coordinates_storage.push_back(coordinates[i]);
                _compressed_locks.push_back(iterator == _user_locks.end() ? false : (*iterator).second[i]);
            }
        }
        return (*res.first).second;
    };
    _tetrahedra.reserve(_mesh.nb_cells());
    _tetrahedron_refs.reserve(_mesh.nb_cells());
    for (auto cell : _mesh.cell_range()) {
        bool hasUnlocked = false;
        auto vertex_indices = _mesh.cell_vertices(cell);
        for (unsigned i = 0; i < 4; ++i) {
            if (!is_vert_locked(vertex_indices[i])) {
                hasUnlocked = true;
                break;
            }
        }
        if (!hasUnlocked) continue;

        auto cell_ref = _mesh.cell_reference_shape(cell);
        std::array<Eigen::Vector3d, 4> cell_ref_eigen;
        std::array<unsigned, 4> compressed_index;
        for (unsigned i = 0; i < 4; ++i) {
            cell_ref_eigen[i] = convert_to_eigen(cell_ref[i]);
            compressed_index[i] = get_compressed_point_id(vertex_indices[i]);
            ++nb_tet_on_verts[compressed_index[i]];
        }
        _cell_original_to_compressed.emplace(cell, _tetrahedra.size());
        _tetrahedra.push_back(compressed_index);
        _tetrahedron_refs.push_back(Mesh_optimization_internal::Math_functions::transform_coordinates_to_gradient_base(cell_ref_eigen));
    }

    _tetrahedra.shrink_to_fit();
    _tetrahedron_refs.shrink_to_fit();
    _compressed_coords = Eigen::Map<Eigen::VectorXd>(temp_coordinates_storage.data(), static_cast<Eigen::Index>(temp_coordinates_storage.size()));

    // computing vertex -> tet data structure
    std::vector<unsigned> curr_id(nb_points,0);
    _vert2tet_corner.reserve(nb_points);
    for (unsigned v = 0; v < nb_points; ++v) {
        _vert2tet_corner.push_back(std::vector<unsigned>(nb_tet_on_verts[v]));
    }
    for (unsigned t = 0; t < _tetrahedra.size(); ++t) {
        for (unsigned i = 0; i < 4; ++i) {
            unsigned v = _tetrahedra[t][i];
            _vert2tet_corner[v][curr_id[v]++] = 4*t+i;
        }
    }
    initialize_boundary();
    initialize_curve_network();
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::initialize_boundary() {
    std::vector<unsigned> currFace;
    _bnd_faces.reserve(_boundary.nb_faces());
    _face_surface_id.reserve(_boundary.nb_faces());
    std::unordered_map<unsigned, std::vector<std::array<unsigned, 2>>> vert2faces; // todo: maybe use a map from boost
    for (auto face : _boundary.face_range()) {
        currFace.clear();
        for (auto vertex_descriptor : _boundary.face_vertices(face)) {
            auto iterator = _vertex_original_to_compressed.find(static_cast<Vertex_descriptor>(vertex_descriptor));
            // it is not dramatic for Tetrahedral_conformal_optimizer to not have the perfect face
            if (iterator == _vertex_original_to_compressed.end()) continue;
            currFace.push_back(iterator->second);
        }
        if (currFace.empty()) continue;
        for (unsigned i = 0; i < currFace.size(); ++i) {
            if (_lock_boundary) {
                for (unsigned d = 0; d < 3; ++d) {
                    _compressed_locks[3*currFace[i]+d] = true;
                }
            }
            std::array<unsigned, 2> location_pair = {static_cast<unsigned>(_bnd_faces.size()), i};
            auto res = vert2faces.emplace(currFace[i], std::vector<std::array<unsigned, 2>>{location_pair});
            if (!res.second) (*res.first).second.push_back(location_pair);
        }
        _bnd_faces.push_back(currFace);
        _face_surface_id.push_back(_boundary.surface_id(face));
    }
    // not ordered, do a sort? then I lose memory alignment? check if becomes bottleneck
    _vert_and_face_corners.assign(vert2faces.begin(), vert2faces.end());
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::initialize_curve_network() {
    std::vector<unsigned> currFace;
    _curve_edges.reserve(_edge_network.nb_edges());
    _curve_ids.reserve(_edge_network.nb_edges());
    for (auto edge : _edge_network.edge_range()) {
        std::array<unsigned, 2> edge_vertices;
        bool not_in = false;
        for (unsigned i = 0; i < 2; ++i) {
            Vertex_descriptor v = _edge_network.edge_vertex(edge, i);
            auto iterator = _vertex_original_to_compressed.find(static_cast<Vertex_descriptor>(v));
            // if we do not find the vertex, we just ignore the edge
            if (iterator == _vertex_original_to_compressed.end()) {
                not_in = true;
                break;
            }
            edge_vertices[i] = iterator->second;
        }
        if (not_in) continue;
        _curve_edges.push_back(edge_vertices);
        _curve_ids.push_back(_edge_network.curve_id(edge));
    }

    _point_targets.reserve(_vertex_target_positions.size());
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::initialise_point_targets() {
    for (auto [v, target] : _vertex_target_positions) {
        auto iterator = _vertex_original_to_compressed.find(static_cast<Vertex_descriptor>(v));
        if (iterator == _vertex_original_to_compressed.end()) continue;
        _point_targets.push_back({iterator->second, convert_to_inner(target)});
    }
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::rescale_geometry() {
    if (_compressed_coords.size() == 0) return;
    std::size_t nb_vertices = static_cast<std::size_t>(_compressed_coords.size()) / 3;
    Eigen::Vector3d mini(_compressed_coords[0], _compressed_coords[1], _compressed_coords[2]);
    Eigen::Vector3d maxi(_compressed_coords[0], _compressed_coords[1], _compressed_coords[2]);
    Eigen::Vector3d center(0.,0.,0.);
    for (unsigned i = 0; i < nb_vertices; ++i) {
        for (unsigned d = 0; d < 3;++d) {
            mini[d] = (std::min)(mini[d], _compressed_coords[3*i+d]);
            maxi[d] = (std::max)(maxi[d], _compressed_coords[3*i+d]);
            center[d] += _compressed_coords[3*i+d];
        }
    }

    center /= static_cast<double>(nb_vertices);
    _scale = (std::max)((std::max)(maxi[0] - mini[0], maxi[1] - mini[1]), maxi[2]-mini[2]) / _rescaled_bbox_scale;
    if (_scale < 1e-14) { // todo figure out a better thresh old here
        _scale = 1.;
        return;
    }
    _shift = center;

    for (int i = 0; i < static_cast<int>(nb_vertices); ++i) {
        for (int d = 0; d < 3;++d) {
            _compressed_coords[3*i+d] = (_compressed_coords[3*i+d] - _shift[d]) / _scale;
        }
    }
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::update_mesh_coordinates() {
    for (auto [vertex_descriptor, compressed_id] : _vertex_original_to_compressed) {
        Eigen::Vector3d pt =Mesh_optimization_internal::Math_functions::sub_col_vector(_compressed_coords, compressed_id);
        _mesh.set_new_vertex_coordinates(vertex_descriptor, convert_to_user(pt));
    }
}


template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::set_callback_function(Callback_function callback_function, Callback_setting setting) {
    _callback_function = callback_function;
    _callback_setting = setting;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::initialize_callback() {
    if (_callback_initialized) return;
    _callback_vertex_map_data.reserve(_vertex_original_to_compressed.size());
    _callback_cell_map_data.reserve(_cell_original_to_compressed.size());
    Vertex_status default_vertex_status;
    for (auto [vertex_descriptor, compressed_id] : _vertex_original_to_compressed) {
        _callback_vertex_map_data.emplace(vertex_descriptor, default_vertex_status);
    }
    Cell_status default_cell_status;
    for (auto [cell_descriptor, compressed_id] : _cell_original_to_compressed) {
        _callback_cell_map_data.emplace(cell_descriptor, default_cell_status);
    }
    _callback_initialized = true;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
bool Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::run_callback(Iteration_status const &status, std::vector<Vertex_status> const &vertex_status, std::vector<Cell_status> const &cell_status) {
    initialize_callback();
    update_mesh_coordinates();
    _callback_status = status;
    _callback_status.scaling_factor = _scale;
    for (auto [vertex_descriptor, compressed_id] : _vertex_original_to_compressed) {
        _callback_vertex_map_data.at(vertex_descriptor) = vertex_status[compressed_id];
    }
    for (auto [cell_descriptor, compressed_id] : _cell_original_to_compressed) {
        _callback_cell_map_data.at(cell_descriptor) = cell_status[compressed_id];
    }
    return _callback_function(_callback_status, _callback_vertex_map_data, _callback_cell_map_data);
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::initialise_boundary_query(Mesh_optimization_internal::Tetrahedral_conformal_optimizer &optimizer) {
    switch (_boundary_query_type) {
    case NONE:
        optimizer.set_boundary_with_singular_query(
            _bnd_faces,
            &_vert_and_face_corners,
            nullptr,
            _face_surface_id
        );
        break;
    case POINT_QUERY:
        {
            Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Boundary_query boundary_query = [&](std::vector<Eigen::Vector3d> const &coords, unsigned surface_id) {
                Eigen::Vector3d center = Eigen::Vector3d::Zero();
                for (auto const &coord : coords) {
                    center += coord;
                }
                center /= static_cast<double>(coords.size());
                double radius = 0;
                for (auto const &coord : coords) {
                    radius += (coord-center).norm();
                }
                radius /= static_cast<double>(coords.size());
                radius /= _scale;
                auto [user_point, user_normal] =  _boundary_point_query(convert_to_user(center), surface_id, radius);
                Eigen::Vector3d proj = convert_to_inner(user_point);
                Eigen::Vector3d normal = convert_to_eigen(user_normal);
                normal.normalize();
                return Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Plane{ proj, normal };
            };
            optimizer.set_boundary_with_singular_query(
                _bnd_faces,
                &_vert_and_face_corners,
                boundary_query,
                _face_surface_id
            );
        }
        break;
    case ELEMENT_QUERY:
        {
            Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Boundary_query boundary_query = [&](std::vector<Eigen::Vector3d> const &coords, unsigned surface_id) {
                std::vector<Point_3> polygon(coords.size());
                for (unsigned i = 0; i < coords.size(); ++i) {
                    polygon[i] = convert_to_user(coords[i]);
                }
                auto [user_point, user_normal] =  _boundary_polygon_query(polygon, surface_id);
                Eigen::Vector3d proj = convert_to_inner(user_point);
                Eigen::Vector3d normal = convert_to_eigen(user_normal);
                normal.normalize();
                return Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Plane{ proj, normal };
            };
            optimizer.set_boundary_with_singular_query(
                _bnd_faces,
                &_vert_and_face_corners,
                boundary_query,
                _face_surface_id
            );
        }
        break;
    case BATCH_POINT_QUERY:
        {
            Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Boundary_batch_query boundary_query = [&](std::vector<std::vector<Eigen::Vector3d>> const &coords, std::vector<unsigned> surface_ids, std::vector<Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Plane> &results) {
                if (coords.size() != _boundary_batch_info_points.size()) {
                    _boundary_batch_info_points.resize(coords.size());
                    _boundary_batch_info_radii.resize(coords.size());
                    _boundary_batch_planes.resize(coords.size());
                }
#pragma omp parallel for
                for (int iter_t = 0; iter_t < static_cast<int>(coords.size()); ++iter_t) {
                    unsigned uiter = static_cast<unsigned>(iter_t);
                    Eigen::Vector3d center = Eigen::Vector3d::Zero();
                    for (auto const &coord : coords[uiter]) {
                        center += coord;
                    }
                    center /= static_cast<double>(coords[uiter].size());
                    double radius = 0;
                    for (auto const &coord : coords[uiter]) {
                        radius += (coord-center).norm();
                    }
                    radius /= static_cast<double>(coords[uiter].size());
                    radius /= _scale;
                    _boundary_batch_info_points[uiter] = convert_to_user(center);
                    _boundary_batch_info_radii[uiter] = radius;
                }

                _boundary_point_batch_query(_boundary_batch_info_points, surface_ids, _boundary_batch_info_radii, _boundary_batch_planes);

#pragma omp parallel for
                for (int iter_t = 0; iter_t < static_cast<int>(coords.size()); ++iter_t) {
                    auto [user_point, user_normal] =  _boundary_batch_planes[static_cast<unsigned>(iter_t)];
                    Eigen::Vector3d proj = convert_to_inner(user_point);
                    Eigen::Vector3d normal = convert_to_eigen(user_normal);
                    normal.normalize();
                    results[static_cast<unsigned>(iter_t)] = { proj, normal };
                }
            };
            optimizer.set_boundary_with_batch_query(
                _bnd_faces,
                &_vert_and_face_corners,
                boundary_query,
                _face_surface_id
            );
        }
        break;
    case BATCH_ELEMENT_QUERY:
        {
            Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Boundary_batch_query boundary_query = [&](std::vector<std::vector<Eigen::Vector3d>> const &coords, std::vector<unsigned> surface_ids, std::vector<Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Plane> &results) {
                if (coords.size() != _boundary_batch_info_polygons.size()) {
                    _boundary_batch_info_polygons.reserve(coords.size());
                    for (unsigned i = 0; i < coords.size(); ++i) {
                        _boundary_batch_info_polygons.push_back(std::vector<Point_3>(coords[i].size()));
                    }
                    _boundary_batch_planes.resize(coords.size());
                }
#pragma omp parallel for
                for (int iter_t = 0; iter_t < static_cast<int>(coords.size()); ++iter_t) {
                    for (unsigned i = 0; i < coords.size(); ++i) {
                        _boundary_batch_info_polygons[static_cast<unsigned>(iter_t)][i] = convert_to_user(coords[static_cast<unsigned>(iter_t)][i]);
                    }
                }

                _boundary_polygon_batch_query(_boundary_batch_info_polygons, surface_ids, _boundary_batch_planes);

#pragma omp parallel for
                for (int iter_t = 0; iter_t < static_cast<int>(coords.size()); ++iter_t) {
                    auto [user_point, user_normal] =  _boundary_batch_planes[static_cast<unsigned>(iter_t)];
                    Eigen::Vector3d proj = convert_to_inner(user_point);
                    Eigen::Vector3d normal = convert_to_eigen(user_normal);
                    normal.normalize();
                    results[static_cast<unsigned>(iter_t)] = { proj, normal };
                }
            };
            optimizer.set_boundary_with_batch_query(
                _bnd_faces,
                &_vert_and_face_corners,
                boundary_query,
                _face_surface_id
            );
        }
        break;
    }
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::initialise_curve_queries(Mesh_optimization_internal::Tetrahedral_conformal_optimizer &optimizer) {
    switch (_curve_query_type) {
    case NONE:
        break;
    case POINT_QUERY:
        {
            Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Curve_query curve_query = [&](std::array<Eigen::Vector3d, 2> const &coords, unsigned curve_id) {
                Eigen::Vector3d center = 0.5*(coords[0] + coords[1]);
                double radius = (coords[1] - coords[0]).norm() / _scale;
                auto [user_point, user_tangent] =  _curve_point_query(convert_to_user(center), curve_id, radius);
                Eigen::Vector3d proj = convert_to_inner(user_point);
                Eigen::Vector3d tangent = convert_to_eigen(user_tangent);
                tangent.normalize();
                return Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Curve_tangent{ proj, tangent };
            };
            optimizer.set_curve_network_with_singular_query(
                _curve_edges,
                _curve_ids,
                curve_query
            );
        }
        break;
    case ELEMENT_QUERY:
        {
            Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Curve_query curve_query = [&](std::array<Eigen::Vector3d, 2> const &coords, unsigned curve_id) {
                std::array<Point_3, 2> edge = { convert_to_user(coords[0]),  convert_to_user(coords[1])};
                auto [user_point, user_tangent] =  _curve_segment_query(edge, curve_id);
                Eigen::Vector3d proj = convert_to_inner(user_point);
                Eigen::Vector3d tangent = convert_to_eigen(user_tangent);
                tangent.normalize();
                return Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Curve_tangent{ proj, tangent };
            };
            optimizer.set_curve_network_with_singular_query(
                _curve_edges,
                _curve_ids,
                curve_query
            );
        }
        break;
    case BATCH_POINT_QUERY:
        {
            Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Curve_batch_query curve_query = [&](std::vector<std::array<Eigen::Vector3d, 2>> const &edges, std::vector<unsigned> curve_ids, std::vector<Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Curve_tangent> &results) {
                if (edges.size() != _curve_batch_info_points.size()) {
                    _curve_batch_info_points.resize(edges.size());
                    _curve_batch_info_radii.resize(edges.size());
                    _curve_batch_tangents.resize(edges.size());
                }
                for (unsigned e = 0; e < edges.size(); ++e) {
                    _curve_batch_info_points[e] = convert_to_user(0.5*(edges[e][0] + edges[e][1]));
                    _curve_batch_info_radii[e] = (edges[e][1] - edges[e][0]).norm() / _scale;
                }

                _curve_point_batch_query(_curve_batch_info_points, _curve_ids, _curve_batch_info_radii, _curve_batch_tangents);

                for (unsigned e = 0; e < edges.size(); ++e) {
                    auto [user_point, user_tangent] =  _curve_batch_tangents[e];
                    Eigen::Vector3d proj = convert_to_inner(user_point);
                    Eigen::Vector3d tangent = convert_to_eigen(user_tangent);
                    tangent.normalize();
                    results[e] = { proj, tangent };
                }
            };
            optimizer.set_curve_network_with_batch_query(
                _curve_edges,
                _curve_ids,
                curve_query
            );
        }
        break;
    case BATCH_ELEMENT_QUERY:
        {
            Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Curve_batch_query curve_query = [&](std::vector<std::array<Eigen::Vector3d, 2>> const &edges, std::vector<unsigned> curve_ids, std::vector<Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Curve_tangent> &results) {
                if (edges.size() != _curve_batch_info_edges.size()) {
                    _curve_batch_info_edges.resize(edges.size());
                    _curve_batch_tangents.resize(edges.size());
                }
                for (unsigned e = 0; e < edges.size(); ++e) {
                    _curve_batch_info_edges[e] = {convert_to_user(edges[e][0]), convert_to_user(edges[e][1])};
                }

                _curve_segment_batch_query(_curve_batch_info_edges, _curve_ids, _curve_batch_tangents);

                for (unsigned e = 0; e < edges.size(); ++e) {
                    auto [user_point, user_tangent] =  _curve_batch_tangents[e];
                    Eigen::Vector3d proj = convert_to_inner(user_point);
                    Eigen::Vector3d tangent = convert_to_eigen(user_tangent);
                    tangent.normalize();
                    results[e] = { proj, tangent };
                }
            };
            optimizer.set_curve_network_with_batch_query(
                _curve_edges,
                _curve_ids,
                curve_query
            );
        }
        break;
    }
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::initialise_optimizer(Mesh_optimization_internal::Tetrahedral_conformal_optimizer &optimizer) {
    initialise_boundary_query(optimizer);
    initialise_curve_queries(optimizer);

    initialise_point_targets();
    optimizer.set_quadratic_target_positions(_point_targets);

    if (_callback_function != nullptr) {
        optimizer.callback_function = [&](Iteration_status const &status, std::vector<Vertex_status> const &vertex_data, std::vector<Cell_status> const &cell_data)
            {
                return run_callback(status, vertex_data, cell_data);
            };
    }
    optimizer.callback_setting = _callback_setting;
    optimizer.verbose = _verbose;
    optimizer.min_valid_edge_size = _min_valid_edge_size / _scale;
    optimizer.boundary_weight = _boundary_weight;
    optimizer.fine_time_logging = _verbose && _tetrahedra.size() > 15'000'000; // todo: add a number of core specific threshold here
}


template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::naive_smooth() {
    check_refs();
    create_compress_sorted_data();
    rescale_geometry();
    Mesh_optimization_internal::Tetrahedral_conformal_optimizer optimizer(_compressed_coords, _compressed_locks, _tetrahedra, _tetrahedron_refs, _vert2tet_corner);

    initialise_optimizer(optimizer);

    optimizer.run_laplacian_gradient_descent(_max_number_of_iteration);
    update_mesh_coordinates();
    _nb_lbfgs_iterations = optimizer.number_of_lbfgs_iter;
}

template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
bool Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::untangle() {
    check_refs();
    create_compress_sorted_data();
    rescale_geometry();
    Mesh_optimization_internal::Tetrahedral_conformal_optimizer optimizer(_compressed_coords, _compressed_locks, _tetrahedra, _tetrahedron_refs, _vert2tet_corner);

    initialise_optimizer(optimizer);

    bool result = optimizer.run_untangling(_max_number_of_iteration);
    update_mesh_coordinates();
    _nb_lbfgs_iterations = optimizer.number_of_lbfgs_iter;

    return result;
}


template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
bool Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::maximize_quality() {
    check_refs();
    create_compress_sorted_data();
    rescale_geometry();
    Mesh_optimization_internal::Tetrahedral_conformal_optimizer optimizer(_compressed_coords, _compressed_locks, _tetrahedra, _tetrahedron_refs, _vert2tet_corner);

    initialise_optimizer(optimizer);

    bool result = optimizer.run_quality_maximization(_max_number_of_iteration);
    update_mesh_coordinates();
    _nb_lbfgs_iterations = optimizer.number_of_lbfgs_iter;

    return result;
}


// template<typename TetrahedralMesh, typename BoundaryMesh, typename EdgeNetwork>
// void Mesh_conformal_optimizer<TetrahedralMesh, BoundaryMesh, EdgeNetwork>::improve_histogram_iteratively(unsigned number_iterative_steps, double locking_threshold) {
//     check_refs();
//     create_compress_sorted_data();
//     rescale_geometry();


//     Mesh_optimization_internal::Tetrahedral_conformal_optimizer::Boundary_query potential_query = [&](Eigen::Vector3d const &coord, unsigned surface_id, double radius) {
//         return run_boundary_query(coord, surface_id, radius);
//     };

//     double worst_quality = 1e100;
//     std::size_t ref_size = _tetrahedra.size();
//     std::size_t curr_size = _tetrahedra.size();


//     auto update_data = [&]() {
//         if (_verbose) Mesh_optimization_internal::Colorized_print("updating inner data", Mesh_optimization_internal::ConsoleTextColor::Green);
//         for (auto [orig, v] : _vertex_original_to_compressed) {
//             if (_compressed_locks[3*v+0] && _compressed_locks[3*v+1] && _compressed_locks[3*v+2]) {
//                 set_vertex_Lock(orig);
//             }
//         }
//         create_compress_sorted_data();
//         rescale_geometry();
//         ref_size = _tetrahedra.size();
//     };

//     for (unsigned iter = 0; iter < number_iterative_steps; ++iter) {
//         if (_verbose) Mesh_optimization_internal::Colorized_print("Histogram improvement iter " + std::to_string(iter), Mesh_optimization_internal::ConsoleTextColor::Green);
//         if (_verbose) Mesh_optimization_internal::Colorized_print("Current number of optimized tetrahedra: " + std::to_string(curr_size), Mesh_optimization_internal::ConsoleTextColor::Green);

//         Mesh_optimization_internal::Tetrahedral_conformal_optimizer optimizer(_compressed_coords, _compressed_locks, _tetrahedra, _tetrahedron_refs, _vert2tet_corner);

//         initialise_optimizer(optimizer);

//         optimizer.run_quality_maximization(20);
//         update_mesh_coordinates();
//         worst_quality = optimizer.get_max_conformal_energy();
//         auto quality = optimizer.get_conformal_energies();

//         if (_verbose) Mesh_optimization_internal::Colorized_print("New worst quality: " + std::to_string(worst_quality), Mesh_optimization_internal::ConsoleTextColor::Green);

//         if (iter == number_iterative_steps-1) break;
//         curr_size = _tetrahedra.size();
//         for (std::size_t t = 0; t < _tetrahedra.size(); ++t) {
//             if (quality[t] > (1-locking_threshold)*worst_quality) {
//                 --curr_size;
//                 for (unsigned v : _tetrahedra[t]) {
//                     for (unsigned d = 0; d < 3; ++d) {
//                         _compressed_locks[3*v+d] = true;
//                     }
//                 }
//             }
//         }
//         if (curr_size == 0) break;
//         update_data();
//         if (ref_size == 0) break;

//     }
// }

}

