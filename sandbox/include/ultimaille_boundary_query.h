#pragma once
#include <ultimaille/helpers/hboxes.h>
#include <ultimaille/surface.h>

#include "ultimaille_distances.h"

namespace UM_extension {

    using Plane = std::tuple<UM::vec3, UM::vec3>;

    template <unsigned max_size>
    struct Plane_rolling_approximation {
        void add(Plane const &p) {
            for (unsigned i = 1; i < curr_size; ++i) {
                history[i] = history[i-1];
            } 
            history[0] = p;
            curr_size = std::min(max_size, curr_size+1);
        }

        double weight(unsigned i) {
            double total = 0.5 * curr_size*(curr_size+1);
            return (curr_size-i)/total;
        } 

        Plane estimate() {
            Plane p = {UM::vec3{0.,0.,0.}, UM::vec3{0.,0.,0.}}; 
            for (unsigned i = 0; i < curr_size; ++i) {
                std::get<0>(p) += weight(i) * std::get<0>(history[i]);
                std::get<1>(p) += weight(i) * std::get<1>(history[i]);
            } 
            double norm = std::get<1>(p).norm();
            std::get<1>(p) = norm > 1e-14 ? std::get<1>(p)/norm : UM::vec3{0.,0.,0.};
            return p;
        } 

        unsigned curr_size = 0;
        std::array<Plane, max_size> history;
    };

    template <unsigned esperance>
    struct Plane_stabilizing_approximation {
        void add(Plane const &p) {
            double w = curr_weight(); 
            std::get<0>(curr_plane) = (1-w) * std::get<0>(curr_plane) + w * std::get<0>(p);  
            std::get<1>(curr_plane) = (1-w) * std::get<1>(curr_plane) + w * std::get<1>(p);  

            double norm = std::get<1>(curr_plane).norm();
            std::get<1>(curr_plane) = norm > 1e-14 ? std::get<1>(curr_plane)/norm : UM::vec3{0.,0.,0.};
            ++curr_nb_samples;
        }

        double curr_weight() {
            double shift = (std::max(curr_nb_samples, esperance) - esperance) + 1.;
            return 1./std::pow(shift, 3./2.);
        } 

        Plane estimate() {
            return curr_plane;
        }
        unsigned curr_nb_samples = 0;
        Plane curr_plane;
    };


    class Surface_projector {
    public:
        Surface_projector(UM::Triangles const &surface_mesh) 
        : _mesh(const_cast<UM::Triangles &>(surface_mesh))
        , _bvh(surface_mesh) 
        , _data(surface_mesh.nfacets())
        { 
            for (auto f : _mesh.iter_facets()) {
                _data[f].n = UM::Triangle3(f).normal();
                _data[f].edge_size = std::sqrt(UM::Triangle3(f).unsigned_area());
            }
            
        }

        UM::vec3 get_closest_point(UM::vec3 const &pt) {
            return _bvh.nearest_point(pt).p;
        }

        std::tuple<UM::vec3, unsigned> get_closest_point_and_entity(UM::vec3 const &pt) {
            auto res = _bvh.nearest_point(pt);
            return {res.p, res.f};
        }

        Plane get_closest_tangent_plane(UM::vec3 const &pt) {
            auto res = _bvh.nearest_point(pt);
            return {res.p, _data[res.f].n};
        }

        Plane get_triangle_dynamic_sampling_projection(std::array<UM::vec3, 3> const &points);

        // splits into triangles
        Plane get_polygon_dynamic_sampling_projection(std::vector<UM::vec3> const &points);


        std::function<std::tuple<UM::vec3, UM::vec3, double> (UM::vec3 const &, unsigned, double)> get_callable_custom_point_query() {
            return [&] (UM::vec3 const &pt, unsigned, double) {
                auto [p, n] = get_closest_tangent_plane(pt);
                return std::tuple<UM::vec3, UM::vec3, double>{p, n, 1.};
            };
        }

        std::function<std::tuple<UM::vec3, UM::vec3, double> (std::vector<UM::vec3> const &, unsigned)> get_callable_custom_polygon_query() {
            return [&] (std::vector<UM::vec3> const &pts, unsigned) {
                auto [p, n] = get_polygon_dynamic_sampling_projection(pts);
                return std::tuple<UM::vec3, UM::vec3, double>{p, n, 1.};
            };
        }
        
    private:

        UM::Triangles &_mesh;
        UM::BVHTriangles _bvh;
        struct Tri_data {
            UM::vec3 n;
            double edge_size;
        };
        std::vector<Tri_data> _data;
    };
}

namespace UM_extension {

    Plane Surface_projector::get_triangle_dynamic_sampling_projection(std::array<UM::vec3, 3> const &points) {
        auto get_coord = [&](double l0, double l1) {
            return points[1]*l0 + points[2]*l1 + points[0]*(1.-l0-l1);
        };
        double edge_size_0 = (points[0] - points[1]).norm();
        double edge_size_1 = (points[0] - points[2]).norm();
        auto [p, closestTri] = get_closest_point_and_entity(get_coord(1./3., 1./3.));

        UM::PolyLine triangle;
        UM::PolyLine result;
        UM::PolyLine samples;

        // static unsigned triangle_id = 0;
        // ++triangle_id;
        // triangle.points.data->push_back(points[0]);
        // triangle.points.data->push_back(points[1]);
        // triangle.points.data->push_back(points[2]);
        // triangle.edges.push_back(0); triangle.edges.push_back(1);
        // triangle.edges.push_back(1); triangle.edges.push_back(2);
        // triangle.edges.push_back(2); triangle.edges.push_back(0);
        // UM::write_by_extension("triangle"+std::to_string(triangle_id)+".geogram", triangle);


        if (_data[closestTri].edge_size > 0.5*std::max(edge_size_0, edge_size_1)) {
            // result.edges.push_back(0); result.points.push_back(p);
            // result.edges.push_back(1); result.points.push_back(p+std::max(edge_size_0, edge_size_1)*_data[closestTri].n);
            // UM::write_by_extension("result"+std::to_string(triangle_id)+".geogram", result);
            return {p, _data[closestTri].n};
        }

        double sampled_edge_size = _data[closestTri].edge_size;
        unsigned nb_samples_edge_0 = static_cast<unsigned>(std::floor(edge_size_0 / sampled_edge_size))+1;
        unsigned nb_samples_edge_1 = static_cast<unsigned>(std::floor(edge_size_1 / sampled_edge_size))+1;

        UM::vec3 avg_proj = p;
        UM::vec3 avg_normal = _data[closestTri].n;
        unsigned nb_samples = 1;


        for (unsigned i = 0; i < nb_samples_edge_0; ++i) {
            double l0 = (i+1) / static_cast<double>(nb_samples_edge_0+1);
            for (unsigned j = 0; j < nb_samples_edge_1; ++j) {
                double l1 = (j+1) / static_cast<double>(nb_samples_edge_1+1);
                if (l0 + l1 + 1e-12 > 1.) continue;
                auto [pp, n] = get_closest_tangent_plane(get_coord(l0, l1));
                avg_proj += pp;
                avg_normal += n;
                ++nb_samples;
                // samples.edges.push_back(samples.nverts()); samples.points.push_back(get_coord(l0, l1));
                // samples.edges.push_back(samples.nverts()); samples.points.push_back(pp);
            }
        }
        avg_proj /= nb_samples;
        double norm = avg_normal.norm();
        avg_normal = norm > 1e-14 ? avg_normal/norm : UM::vec3{0.,0.,0.};
        
        // result.edges.push_back(0); result.points.push_back(avg_proj);
        // result.edges.push_back(1); result.points.push_back(avg_proj+std::max(edge_size_0, edge_size_1)*avg_normal);
        // UM::write_by_extension("result"+std::to_string(triangle_id)+".geogram", result);
        // UM::write_by_extension("samples"+std::to_string(triangle_id)+".geogram", samples);

        return {avg_proj, avg_normal};
    }

    Plane Surface_projector::get_polygon_dynamic_sampling_projection(std::vector<UM::vec3> const &points) {
        if (points.size() == 1) {
            return get_closest_tangent_plane(points[0]);
        }
        if (points.size() == 2) {
            return get_triangle_dynamic_sampling_projection({points[0], points[1], points[1]});
        }
        if (points.size() == 3) {
            return get_triangle_dynamic_sampling_projection({points[0], points[1], points[2]});
        }

        UM::vec3 center = {0,0,0};
        for (auto const &p : points) center = center + p;
        center = center / static_cast<double>(points.size());

        UM::vec3 avg_normal;
        UM::vec3 avg_proj;
        for (unsigned i = 0; i < points.size(); ++i) {
            auto [p, n] = get_triangle_dynamic_sampling_projection({points[i], points[(i+1)%points.size()], center});
            avg_normal += n;
            avg_proj += p;
        }
        double norm = avg_normal.norm();
        avg_normal = norm > 1e-14 ? avg_normal/norm : UM::vec3{0.,0.,0.};
        avg_proj = avg_proj / static_cast<double>(points.size());
        return {avg_proj, avg_normal};
    }

}
