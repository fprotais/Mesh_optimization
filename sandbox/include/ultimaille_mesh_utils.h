#pragma once
#include <ultimaille/all.h>
#include <unordered_set>

namespace UM_extension {
    template <typename MeshType>
    void clean_mesh(MeshType &mesh) {
        mesh.points.data->clear();
        mesh.cells.clear();
    }

    template <typename MeshType>
    void clean_surface(MeshType &mesh) {
        mesh.points.data->clear();
        mesh.facets.clear();
    }

    template <typename MeshType>
    void copy_mesh(MeshType const &original, MeshType &copy, std::vector<bool> const &cell_to_kill = {}) {
        clean_mesh(copy);
        copy.points.data->assign(original.points.begin(), original.points.end());
        copy.cells.assign(original.cells.begin(), original.cells.end());
        if (cell_to_kill.empty()) return;
        copy.delete_cells(cell_to_kill);
        copy.delete_isolated_vertices();
    }

    template <typename MeshType, typename Query>
    void copy_mesh_with_query(MeshType const &original, MeshType &copy, Query cell_to_keep) {
        std::vector<bool> tokill(original.ncells());
        for (int c=0; c<original.ncells(); ++c) {
            tokill[c] = !cell_to_keep(c); 
        }
        copy_mesh(original, copy, tokill);
    }

    template <typename MeshType>
    void copy_surface(MeshType const &original, MeshType &copy) {
        copy.points.data->assign(original.points.begin(), original.points.end());
        copy.facets.assign(original.facets.begin(), original.facets.end());
    }

    
    template <typename M1, typename M2>
    void rescale_to_same_bbox(M1 &ref, M2 &changed, double coeff = 1.0) {
        UM::BBox3 box_ref = UM::Inspect(ref.points).bbox();
        UM::BBox3 box_changed = UM::Inspect(changed.points).bbox();
        UM::vec3 center_mesh = 0.5 * (box_ref.min + box_ref.max);
        UM::vec3 center_boundary = 0.5 * (box_changed.min + box_changed.max);
        double scale = coeff * std::max(box_ref.max.x - box_ref.min.x, std::max(box_ref.max.y - box_ref.min.y, box_ref.max.z - box_ref.min.z))
                        / std::max(box_changed.max.x - box_changed.min.x, std::max(box_changed.max.y - box_changed.min.y, box_changed.max.z - box_changed.min.z));
        for (auto &p : changed.points) {
            p = scale * (p - center_boundary) + center_mesh;
        }
    }

    template <typename VolumeMeshType, typename SurfaceMeshType, typename Query>
    void extract_interface(VolumeMeshType &mesh, SurfaceMeshType &surface, Query is_inside, std::vector<bool> *external_boundary_vertex = nullptr) {
        mesh.connect();
        surface.facets.clear();
        if (external_boundary_vertex != nullptr) external_boundary_vertex->clear();
        if (external_boundary_vertex != nullptr) external_boundary_vertex->resize(mesh.nverts(), false);

        surface.points.data->assign(mesh.points.begin(), mesh.points.end());
        for (auto f : mesh.iter_facets()) {
            if  (f.on_boundary() && external_boundary_vertex != nullptr) {
                for (int j=0; j < static_cast<int>(f.size()); ++j) {
                    (*external_boundary_vertex)[f.vertex(j)] = true;
                }
                if (is_inside(f.cell())) {
                    for (int j=0; j < static_cast<int>(f.size()); ++j) {
                        surface.facets.push_back(f.vertex(j));
                    }
                }
            }
            else if (is_inside(f.cell()) && !is_inside(f.opposite().cell()))  {
                for (int j=0; j < static_cast<int>(f.size()); ++j) {
                    surface.facets.push_back(f.vertex(j));
                }
            }
        }
    }


    void padd_hex_around_vertex(
        UM::Hexahedra &mesh, 
        unsigned center_vertex, 
        std::vector<std::unordered_set<int>> & vertex_cells, 
        std::map<std::set<int>, std::array<int, 2>> &cell_adjacency_map,
        UM::CellAttribute<int> * region = nullptr
    ) {
        std::unordered_map<int, int> new_vertex;
        std::vector<std::array<int, 9>> new_hexes;

        auto cells = vertex_cells[center_vertex];

        auto opp_cell = [&](UM::Hexahedra::Facet const &f) {
            std::set<int> face = {f.vertex(0), f.vertex(1), f.vertex(2), f.vertex(3)};
            auto res = cell_adjacency_map.find(face);
            if (res == cell_adjacency_map.end()) return -1;
            return res->second[0] == f.cell() ? res->second[1] : res->second[0];
        };

        for (auto cell_id : cells) {
            UM::Hexahedra::Cell c(mesh, cell_id);
            for (auto f : c.iter_facets()) {
                auto opp_cell_id = opp_cell(f);
                if (opp_cell_id == -1) continue;
                auto opp_c = UM::Hexahedra::Cell(mesh, opp_cell_id);
                if (cells.contains((int)opp_c)) continue;
                for (unsigned i = 0; i < 4; ++i) {
                    auto res = new_vertex.emplace(f.vertex(i), mesh.nverts());
                    if (!res.second) continue;
                    vertex_cells.push_back({}); 
                    mesh.points.push_back(0.7*mesh.points[center_vertex] + 0.3* mesh.points[f.vertex(i)]);

                }
                new_hexes.push_back(std::array<int,9>{
                    new_vertex.at(f.vertex(0)),
                    new_vertex.at(f.vertex(1)),
                    new_vertex.at(f.vertex(3)),
                    new_vertex.at(f.vertex(2)),
                    f.vertex(0), 
                    f.vertex(1), 
                    f.vertex(3), 
                    f.vertex(2),
                    region == nullptr ? 0 : (*region)[c]
                });
            }
        }
        
        auto remove = [&](UM::Hexahedra::Facet const &f) {
            std::set<int> face = {f.vertex(0), f.vertex(1), f.vertex(2), f.vertex(3)};
            auto res = cell_adjacency_map.find(face);
            if (res == cell_adjacency_map.end()) return;
            unsigned id = res->second[0] == f.cell() ? 0 : 1;
            res->second[id] = -1;
            if (res->second[(id+1)%2] == -1) {
                cell_adjacency_map.erase(face);
            }
        };
        auto insert = [&](UM::Hexahedra::Facet const &f) {
            std::set<int> face = {f.vertex(0), f.vertex(1), f.vertex(2), f.vertex(3)};
            auto res = cell_adjacency_map.emplace(face, std::array<int, 2>{f.cell(),0});
            if (res.second) return;
            unsigned id = res.first->second[0] == -1 ? 0 : 1;
            res.first->second[id] = f.cell();
        };

        for (auto cell_id : cells) {
            UM::Hexahedra::Cell c(mesh, cell_id);
            for (auto f : c.iter_facets()) {
                remove(f);
            }
            for (unsigned i = 0; i < 8; ++i) {
                auto iter = new_vertex.find(c.vertex(i));
                if (iter == new_vertex.end()) continue;
                vertex_cells[mesh.vert(c, i)].erase(cell_id);
                vertex_cells[iter->second].insert(cell_id);
                mesh.vert(c, i) = iter->second;
            }
            for (auto f : c.iter_facets()) {
                insert(f);
            }
        }
        unsigned start = mesh.create_cells(new_hexes.size());
        for (unsigned h=0; h<new_hexes.size(); ++h) {
            for (unsigned i=0; i<8; ++i) {
                mesh.vert(start + h, i) = new_hexes[h][i];
                vertex_cells[new_hexes[h][i]].insert(start + h);
            }
            if (region != nullptr) (*region)[start+h] = new_hexes[h][8];
            for (auto f : UM::Hexahedra::Cell(mesh, start+h).iter_facets()) {
                insert(f);
            }
        }
        
    }

    // different padding location that above func
    template <typename IsInside>
    void padd_hexmesh(UM::Hexahedra &mesh, IsInside is_inside, UM::CellAttribute<int> * region = nullptr, bool both_side = true) {
        std::vector<std::array<int, 2>> new_vertex(mesh.nverts(), {-1,-1});

        std::vector<std::array<int, 9>> new_hexes;

        for (auto c : mesh.iter_cells()) {
            if (!is_inside(c)) continue;
            for (auto f : c.iter_facets()) {
                auto opp_f  = f.opposite();
                if (opp_f == -1) continue;
                auto opp_c = opp_f.cell();
                if (is_inside(opp_c)) continue;
                for (unsigned i = 0; i < 4; ++i) {
                    for (unsigned d = 0; d < 2; ++d) {
                        if (!both_side && d==0) continue;
                        if (new_vertex[f.vertex(i)][d] != -1) continue;
                        new_vertex[f.vertex(i)][d] = mesh.nverts();
                        mesh.points.push_back(mesh.points[f.vertex(i)]);
                    }
                }
                new_hexes.push_back(std::array<int,9>{
                    new_vertex[f.vertex(0)][1],
                    new_vertex[f.vertex(1)][1],
                    new_vertex[f.vertex(3)][1],
                    new_vertex[f.vertex(2)][1],
                    f.vertex(0), 
                    f.vertex(1), 
                    f.vertex(3), 
                    f.vertex(2),
                    region == nullptr ? 0 : (*region)[c]
                });
                if (!both_side) continue;

                new_hexes.push_back(std::array<int,9>{
                    f.vertex(0), 
                    f.vertex(1), 
                    f.vertex(3), 
                    f.vertex(2),
                    new_vertex[f.vertex(0)][0],
                    new_vertex[f.vertex(1)][0],
                    new_vertex[f.vertex(3)][0],
                    new_vertex[f.vertex(2)][0],
                    region == nullptr ? 0 : (*region)[opp_c]
                });

            }
        }

        std::vector<UM::vec3> new_coordinates(mesh.nverts(), {0.,0.,0.});
        std::vector<double> weight(mesh.nverts(), 0);
        for (auto he : mesh.iter_halfedges()) {
            // if (new_vertex[he.from()][is_inside(he.cell())] == -1) continue;
            int v = new_vertex[he.to()][is_inside(he.cell())];
            if (v == -1) continue;
            weight[v] += 1;
            new_coordinates[v] += mesh.points[he.from()];
            // new_coordinates[v] += mesh.points[he.from()];
        }
        for (auto v : mesh.iter_vertices()) {
            if (weight[v] == 0) continue;
            mesh.points[v] = new_coordinates[v] / weight[v];
        }



        mesh.disconnect();
        for (auto c : mesh.iter_cells()) {
            for (unsigned i = 0; i < 8; ++i) {
                if (new_vertex[c.vertex(i)][is_inside(c)] == -1) continue;
                mesh.vert(c, i) = new_vertex[c.vertex(i)][is_inside(c)];
            }
        }
        unsigned start = mesh.create_cells(new_hexes.size());
        for (unsigned h=0; h<new_hexes.size(); ++h) {
            for (unsigned i=0; i<8; ++i) {
                mesh.vert(start + h, i) = new_hexes[h][i];
            }
            if (region != nullptr) (*region)[start+h] = new_hexes[h][8];
        }
        
        mesh.connect();
    }

}