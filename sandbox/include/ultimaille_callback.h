#pragma once
#include <ultimaille/all.h>
#include <Mesh_smoothing_3/Mesh_smoothing_3.h>
#include "ultimaille_interfaces.h"
#include "ultimaille_mesh_utils.h"


namespace UM_extension {
    using Smoother = Mesh_smoothing_3::Mesh_smoother<UM_extension::Tetrahedral_mesh_wrapper, UM_extension::Triangle_boundary_wrapper>;
    using Callback_setting = Smoother::Callback_setting;

    template <typename MeshType>
    class Callback_structure {
    public:

        Callback_structure(MeshType const&mesh, std::string mesh_name_ = "mesh") 
        : mesh_name(mesh_name_)
        , m(mesh)
        , locks(m.points)
        , local_edge_size(m.points)
        , gradient_norm(m.points)
        , smoothing_gradient(m.points)
        , boundary_gradient(m.points)
        , move_gradient(m.points)
        , bfgs_gradient(m.points)
        , cell_energy(m)
        , cell_det(m)
        , cell_weight(m)
        , cell_selected(m, 1)
        , cell_inverted(m, 1)
        {
            init_vec_mesh(smoothing_gradient_mesh);
            init_vec_mesh(boundary_gradient_mesh);
        }

        unsigned bfgs_max_display_iter = 1500;
        unsigned bfgs_iter_frequency = 1;
        bool display_gradient = false;

        bool save_video = false;
        unsigned video_frequency = 5;
        unsigned curr_count = 0;

        std::string extension = ".geogram";
        std::string mesh_name = "mesh";

        std::function<bool(
            Smoother::Iteration_status const &,
            Smoother::Vertex_descriptor_map<Smoother::Vertex_status> const &,
            Smoother::Cell_descriptor_map<Smoother::Cell_status> const &
        )> get_callable_function() {
            return [&](
                Smoother::Iteration_status const &status,
                Smoother::Vertex_descriptor_map<Smoother::Vertex_status> const &vertex_data,
                Smoother::Cell_descriptor_map<Smoother::Cell_status> const &cell_data
            ) {
                return run(status, vertex_data, cell_data);
            };
        }

        bool save_selected = false;
        bool save_inverted = false;

        std::function<bool(unsigned cell)> cell_selector = nullptr;
        template<typename F>
        void set_selective_display(F f) {
            cell_selector = f;
            save_selected = true;
        }

    public:
        MeshType const&m;

        UM::PointAttribute<bool> locks;
        UM::PointAttribute<double> local_edge_size;
        UM::PointAttribute<double> gradient_norm;
        UM::PointAttribute<UM::vec3> smoothing_gradient;
        UM::PointAttribute<UM::vec3> boundary_gradient;
        UM::PointAttribute<UM::vec3> move_gradient;
        UM::PointAttribute<UM::vec3> bfgs_gradient;
        UM::CellAttribute<double> cell_energy;
        UM::CellAttribute<double> cell_det;
        UM::CellAttribute<double> cell_weight;
        UM::CellAttribute<int> cell_selected;
        UM::CellAttribute<int> cell_inverted;

        MeshType copy_selective_m;
        MeshType copy_inverted_m;

        UM::PolyLine smoothing_gradient_mesh;
        UM::PolyLine boundary_gradient_mesh;

        inline void init_vec_mesh(UM::PolyLine &vec_mesh) {
            for (int i = 0; i < m.nverts(); ++i) {
                vec_mesh.edges.push_back(vec_mesh.nverts());
                vec_mesh.points.data->push_back(m.points[i]);
                vec_mesh.edges.push_back(vec_mesh.nverts());
                vec_mesh.points.data->push_back(m.points[i]);
            }
        };

        inline void update_vec_mesh(UM::PolyLine &vec_mesh, UM::PointAttribute<UM::vec3> const &normal) {
            for (int i = 0; i < m.nverts(); ++i) {
                vec_mesh.points[2*i+0] = m.points[i];
                vec_mesh.points[2*i+1] = m.points[i] + normal[i];
            }
        }

        inline bool run(
            Smoother::Iteration_status const &status,
            Smoother::Vertex_descriptor_map<Smoother::Vertex_status> const &vertex_data,
            Smoother::Cell_descriptor_map<Smoother::Cell_status> const &cell_data
        ) {
            if (status.is_in_lbfgs()) {
                if (status.lbfgs_status.iter > bfgs_max_display_iter)   return false;
                if (status.lbfgs_status.iter % bfgs_iter_frequency != 0)   return false;
            }
            double scale = status.scaling_factor;
            if (!save_video) {
                std::cout << "========================================================= " << std::endl;
                std::cout << "status.outer_iter_nb: " << status.outer_iter_nb << std::endl;
                std::cout << "status.smoothing_energy: " << status.smoothing_energy << std::endl;
                std::cout << "status.scaling_factor: " << status.scaling_factor << std::endl;
                std::cout << "status.is_in_lbfgs(): " << status.is_in_lbfgs() << std::endl;
                std::cout << "status.lbfgs_status.iter: " << status.lbfgs_status.iter << std::endl;
                std::cout << "status.lbfgs_status.step: " << status.lbfgs_status.step << std::endl;
                std::cout << "status.lbfgs_status.nbEval: " << status.lbfgs_status.nbEval << std::endl;
            }
            double gnorm = 0;
            for (auto [v, vertex_datum] : vertex_data) {
                locks[v] = vertex_datum.is_locked();
                local_edge_size[v] = vertex_datum.local_edge_size;
                smoothing_gradient[v] = UM_extension::eigen2ultimaille(scale * vertex_datum.smoothing_gradient);
                boundary_gradient[v] = UM_extension::eigen2ultimaille(scale * vertex_datum.boundary_gradient);
                move_gradient[v] = UM_extension::eigen2ultimaille(scale * (vertex_datum.smoothing_gradient+vertex_datum.boundary_gradient));
                bfgs_gradient[v] = UM_extension::eigen2ultimaille(scale * vertex_datum.lbfgs_gradient);
                gradient_norm[v] = move_gradient[v].norm();
                gnorm += vertex_datum.lbfgs_gradient.squaredNorm();
            }
            cell_energy.fill(0.);
            cell_weight.fill(0.);
            cell_det.fill(std::numeric_limits<double>::max());
            for (auto [t, cell_datum] : cell_data) {
                if (m.nverts_per_cell() == 4) {
                    cell_energy[t] = std::log10(cell_datum.energy_value);
                    cell_weight[t] = std::log10(cell_datum.weight);
                    cell_det[t] = cell_datum.det;
                }
                else if (m.nverts_per_cell() == 8) {
                    unsigned hex_id = t/8;
                    cell_energy[hex_id] = std::max(cell_energy[hex_id], std::log10(cell_datum.energy_value));
                    cell_weight[hex_id] += std::log10(cell_datum.weight) / 8;
                    cell_det[hex_id] = std::min(cell_det[hex_id], cell_datum.det);
                }

            }

            std::string name_extension;
            if (status.is_in_lbfgs()) name_extension=std::string("_" + std::to_string(status.outer_iter_nb) + "_" + std::to_string(status.lbfgs_status.iter) +extension);
            else name_extension=std::string("_outer_" + std::to_string(status.outer_iter_nb) + extension);

            if (display_gradient) {
                update_vec_mesh(smoothing_gradient_mesh, smoothing_gradient);
                update_vec_mesh(boundary_gradient_mesh, boundary_gradient);
                UM::write_by_extension(mesh_name+"_gradient_boundary" + name_extension, boundary_gradient_mesh);
                UM::write_by_extension(mesh_name+"_gradient_smoothing" + name_extension, smoothing_gradient_mesh);
                update_vec_mesh(smoothing_gradient_mesh, bfgs_gradient);
                UM::write_by_extension(mesh_name+"_gradient_bfgs" + name_extension, smoothing_gradient_mesh);
            }

            for (int c=0; c<m.ncells(); ++c) {
                if (cell_selector) cell_selected[c] = cell_selector(c);
                cell_inverted[c] = cell_det[c]<=0;
            }

            if (save_video) { 
                if (curr_count % video_frequency == 0) {
                    unsigned id = curr_count / video_frequency; 
                    std::cout << "Saving video frame " << id << "(" << curr_count << ")" << std::endl;
                    UM::write_by_extension(mesh_name + "_video_" + std::to_string(id) + extension, m, {{{"locks", locks.ptr}, {"edge_size", local_edge_size.ptr}, {"gradient_norm", gradient_norm.ptr}},
                                                    {{"det", cell_det.ptr}, {"weight_log", cell_weight.ptr}, {"energy", cell_energy.ptr}, {"inside", cell_selected.ptr}, {"inverted", cell_inverted.ptr}}, {{}}, {{}}});
                    if (save_selected && cell_selector) {
                        copy_mesh_with_query(m, copy_selective_m, cell_selector);
                        UM::write_by_extension(mesh_name + "_video_in_" + std::to_string(id) + extension, copy_selective_m);
                    }
                }
                ++curr_count;
                return false;
            }

            if (save_selected && cell_selector) {
                copy_mesh_with_query(m, copy_selective_m, cell_selector);
                UM::write_by_extension(mesh_name + "_inside" + name_extension, copy_selective_m);
            }


            if (save_inverted) {
                copy_mesh_with_query(m, copy_inverted_m, [&](unsigned c) { return cell_selected[c] && cell_inverted[c];});
                UM::write_by_extension(mesh_name + "_inverted" + name_extension, copy_inverted_m);
            }

            std::cout << "Writing: "  << mesh_name + name_extension << "..." << std::endl;
            UM::write_by_extension(mesh_name + name_extension, m, {{{"locks", locks.ptr}, {"edge_size", local_edge_size.ptr}, {"gradient_norm", gradient_norm.ptr}},
                                                                {{"det", cell_det.ptr}, {"weight_log", cell_weight.ptr}, {"energy", cell_energy.ptr}, {"inside", cell_selected.ptr}, {"inverted", cell_inverted.ptr}}, {{}}, {{}}});
            
            
            

            return false;
        }

    };


}

