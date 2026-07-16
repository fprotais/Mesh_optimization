#include <cstdlib>
#include <random>
#include <set>
#include <map>

#include <Mesh_optimization/Mesh_conformal_optimizer.h>

#include "include/ultimaille_interfaces.h"
#include "include/ultimaille_boundary_query.h"
#include "include/ultimaille_mesh_utils.h"
#include <ultimaille/all.h>

class Surface_mesh_wrapper : public Mesh_optimization::helper_structures::Mixed_element_mesh<int, int, UM::vec3, UM_extension::Contiguous_unsigned_range> {
public:
    std::size_t nb_vertices() const override { return mesh.nverts() + scaffold_points.size(); }

    Point_3 vertex_coordinates(Vertex_descriptor vertex) const override {
        return vertex < mesh.nverts() ? mesh.points[vertex] : scaffold_points[vertex - mesh.nverts()];
    }
    void set_new_vertex_coordinates(int vertex, Point_3 coord) override { 
        if (vertex < mesh.nverts()) mesh.points[vertex] = coord;
        else scaffold_points[vertex - mesh.nverts()] = coord;
    }

    UM_extension::Contiguous_unsigned_range input_cell_range() const override { return UM_extension::Contiguous_unsigned_range{0, size_t(2*mesh.nfacets())}; }

    Vertex_descriptor get_cell_vertex(int cell, unsigned local_Vertex_descriptor) const override {
        int face = cell/2;
        if ((int)local_Vertex_descriptor < mesh.facet_size(face)) return mesh.facet(face).vertex(local_Vertex_descriptor);
        return mesh.nverts() + cell;
    };

    Shape const * get_element_shape(int cell) const override {
        int face = cell/2;
        int orientation = cell%2;
        if (mesh.facet_size(face) == 3) return orientation ? &inv_tet_ref : &tet_ref;
        // if (mesh.facet_size(face) == 3) return orientation ? &inv_we_ref : &we_ref;
        else if (mesh.facet_size(face) == 4) return orientation ? &inv_py_ref : &py_ref;
        return nullptr;
    }



public:
    Surface_mesh_wrapper(UM::Polygons &mesh_)
    : mesh(mesh_)
    , scaffold_points(mesh.nfacets() * 2)
    {
        update_scaffold();
        inv_tet_ref.inverse = true;
        inv_py_ref.inverse = true;
        this->assemble(); // CRITICAL
    }

    void update_scaffold() {
        for (auto f : mesh.iter_facets()) {
            UM::vec3 center = UM::Poly3(f).bary_verts();
            UM::vec3 normal = UM::Poly3(f).normal();
            double target_edge = std::sqrt(UM::Poly3(f).unsigned_area());
            scaffold_points[2*f+0] = center + target_edge * normal;
            scaffold_points[2*f+1] = center - target_edge * normal;
        }
    }

    void get_scaffold_meshes(UM::Tetrahedra& tets, UM::Pyramids &pyramids) {
        tets.points.resize(nb_vertices());
        pyramids.points.resize(nb_vertices());
        for (unsigned i = 0; i < nb_vertices(); ++i) {
            tets.points[i] = vertex_coordinates(i);
            pyramids.points[i] = vertex_coordinates(i);
        }
        tets.cells.clear();
        pyramids.cells.clear();
        for (auto f : mesh.iter_facets()) {
            for (unsigned d = 0; d < 2; ++d) {
                for (int i = 0; i < f.size(); ++i) {
                    if (f.size() == 3) tets.cells.push_back(f.vertex(i));
                    else pyramids.cells.push_back(f.vertex(i));
                }
                if (f.size() == 3) tets.cells.push_back(mesh.nverts() + 2*f+d);
                else pyramids.cells.push_back(mesh.nverts() + 2*f+d);
            }
        }
    }
    UM::Polygons &mesh;

    std::vector<UM::vec3> scaffold_points;

    Mesh_optimization::Shapes::VTK_TETRAHEDRON<UM::vec3> tet_ref;
    Mesh_optimization::Shapes::VTK_TETRAHEDRON<UM::vec3> inv_tet_ref;
    Mesh_optimization::Shapes::VTK_PYRAMID<UM::vec3> py_ref;
    Mesh_optimization::Shapes::VTK_PYRAMID<UM::vec3> inv_py_ref;

};

class Boundary_wrapper {
public:
    using Face_descriptor = int;
    using Normal_3 = UM::vec3;
    using Surface_patch_index = unsigned;
    unsigned nb_faces() const { return mesh.nfacets(); }

    UM_extension::Contiguous_unsigned_range face_range() const { return UM_extension::Contiguous_unsigned_range{0, size_t(mesh.nfacets())}; }

    unsigned patch_id(Face_descriptor f) const { return f; }
    unsigned nb_face_vertices(Face_descriptor face) const { return mesh.facet(face).size(); }
    auto face_vertices(Face_descriptor face) const { 
        std::vector<int> vertices(mesh.facet(face).size());
        for (int i = 0; i < mesh.facet(face).size(); ++i) {
            vertices[i] = mesh.facet(face).vertex(i);
        }
        return vertices; 
    }
public:
    UM::Polygons &mesh;
};

class Edge_network {
public:
    using Edge_descriptor = int;
    using Curve_index = unsigned;
    size_t nb_edges() const { return (size_t)selected_edges.nedges(); }
    UM_extension::Contiguous_unsigned_range edge_range() const { return UM_extension::Contiguous_unsigned_range{0, nb_edges()}; }
    unsigned curve_id(int e) const { return (unsigned)e; }
    int edge_vertex(int edge, unsigned i) const { return selected_edges.vert(edge, i); }

public: 
    UM::PolyLine const &selected_edges;
};

class Target_edge_network : UM::HBoxes<3>{
public:
    Target_edge_network(UM::PolyLine & edges_) 
    : m(edges_)
    {
        std::vector<UM::BBox3> bboxes(m.nedges());
        for (int f=0; f<m.nedges(); f++)               // create boxes bounding
            for (int lv=0; lv<2; lv++)    
                bboxes[f].add(m.points[m.vert(f, lv)]);
        init(bboxes);   
    }

    inline double dist_segment(double a, double b, double x) {
            return x < a ? a-x : (x > b ? x-b : 0.);
    }

    inline double dist2_box(const UM::BBox3 &box, const UM::vec3 &p) {
        return UM::vec3(
                dist_segment(box.min.x, box.max.x, p.x),
                dist_segment(box.min.y, box.max.y, p.y),
                dist_segment(box.min.z, box.max.z, p.z)
                ).norm2();
    }


    std::tuple<UM::vec3, UM::vec3> proj(UM::vec3 p, unsigned edge) { // taken from bvh.h
        double best_dist2 = std::numeric_limits<double>::max();
        UM::PolyLine::Edge best_edge = {m, (int) edge};
        UM::vec3 best_point;
        using QEl = std::pair<double, int>;
        std::priority_queue<QEl, std::vector<QEl>, std::greater<QEl>> Q;
        Q.emplace(0., 0);

        while (!Q.empty() && Q.top().first < best_dist2) {
            const int node = Q.top().second; Q.pop();
            const int leaves = tree.size()  - m.nedges();               // start offset for the leaves of the hierarchy
            const int beg = 2*node + 1;                                  // start offset for the children nodes
            const int end = std::min(                                    //   end offset for the children nodes
                    2*node + 3,
                    static_cast<int>(tree.size())
                    );

            for (int son = beg; son<end; son++) {                        // iterate through children boxes
                if (son < leaves)                                        // if it is not a leaf, place it in the priority queue
                    Q.emplace(dist2_box(tree[son], p), son);
                else {
                    UM::PolyLine::Edge e = {m, tree_pos_to_org[son-leaves]}; // for the leaves we can directly compute
                    UM::vec3 nearest = UM::Segment3(e).nearest_point(p);        // the nearest point and compare it to the current best
                    double dist2 = (p-nearest).norm2();
                    if (best_dist2 > dist2) {
                        best_dist2 = dist2;
                        best_edge = e;
                        best_point = nearest;
                    }
                }
            }
        }
        return {best_point, UM::Segment3(best_edge).vector()};
    }
    UM::PolyLine &m;
};

int main(int argc, char** argv) {
    const std::string surface_name = (argc > 1) ? argv[1] : "../data/fandisk_hexalab_kenshi_surf.mesh";
    const std::string target_name = (argc > 2) ? argv[2] : "";

    std::string identifier = "surface_smoothing";


    UM::Polygons mesh;
    UM::read_by_extension(surface_name, mesh);
    mesh.connect();

    UM::Polygons target;
    if (target_name == "") {
        UM::read_by_extension(surface_name, target);
    }
    else {
        UM::read_by_extension(target_name, target);
    }

    target.delete_isolated_vertices();

    bool is_2d_case = true;
    for (auto v: mesh.iter_vertices()) {
        if (UM::vec3(v)[2] != 0) {
            is_2d_case = false;
            break;
        }
    }

    // bool prevent_triangle_intersections = !is_2d_case;

    bool is_open_mesh = false;
    UM::PolyLine boundary_loop;

    for (auto he : mesh.iter_halfedges()) {
        if (he.opposite() == -1) {
            if (!is_open_mesh) boundary_loop.points = mesh.points;
            is_open_mesh = true;
            boundary_loop.edges.push_back(he.from());
            boundary_loop.edges.push_back(he.to());
        }
    } 

    UM::PolyLine target_boundary_loop;
    if (is_open_mesh) {
        target.connect();
        target_boundary_loop.points = target.points;
        for (auto he : target.iter_halfedges()) {
            if (he.opposite() == -1) {
                target_boundary_loop.edges.push_back(he.from());
                target_boundary_loop.edges.push_back(he.to());
            }
        } 
    }
    

    UM::Triangles tri_mesh;
    tri_mesh.points.data->reserve(target.nfacets() * 3);
    tri_mesh.facets.reserve(target.nfacets());
    for (auto f : target.iter_facets()) {
        if (f.size() == 3) {
            for (int i = 0; i < 3; ++i) {
                tri_mesh.facets.push_back(tri_mesh.nverts());
                tri_mesh.points.data->push_back(UM::vec3(f.vertex(i)));
            }
        }
        else if (f.size() == 4) {
            UM::vec3 center = UM::Poly3(f).bary_verts();
            for (unsigned j = 0; j < 4; ++j) {
                for (int i = 0; i < 3; ++i) {
                    tri_mesh.facets.push_back(tri_mesh.nverts() + i);
                }
                tri_mesh.points.data->push_back(UM::vec3(f.vertex(j)));
                tri_mesh.points.data->push_back(UM::vec3(f.vertex((j+1)%f.size())));
                tri_mesh.points.data->push_back(center);
            } 
        }
    }
    UM_extension::Surface_projector surf_proj(tri_mesh);


    UM::write_by_extension(identifier + "_input.mesh", mesh);
    UM::write_by_extension(identifier + "_target.mesh", tri_mesh);

    if (is_open_mesh) {
        UM::write_by_extension(identifier + "_input_curves.mesh", boundary_loop);
        UM::write_by_extension(identifier + "_target_curves.mesh", target_boundary_loop);
    }

    Surface_mesh_wrapper mesh_wrapper(mesh);
    Boundary_wrapper boundary_wrapper {mesh};
    Edge_network curves_wrapper {boundary_loop};
    Target_edge_network target_curves(target_boundary_loop);

    UM::Tetrahedra scaffold_tet; 
    UM::Pyramids scaffold_pyramids;

    mesh_wrapper.get_scaffold_meshes(scaffold_tet, scaffold_pyramids);

    UM::write_by_extension(identifier + "_scaffold_t.mesh", scaffold_tet);
    UM::write_by_extension(identifier + "_scaffold_p.mesh", scaffold_pyramids);


    Mesh_optimization::Mesh_conformal_optimizer optimizer(mesh_wrapper, boundary_wrapper, curves_wrapper);
    optimizer.set_boundary_query(surf_proj.get_callable_custom_point_query());

    if (is_open_mesh) {
        optimizer.set_curves_query([&](UM::vec3 pt, unsigned e, double) {
            auto [proj, normal] = target_curves.proj(pt, e);
            return std::tuple<UM::vec3, UM::vec3, double>{proj, normal, 1.};
        });
    }

    if (is_2d_case) {
        for (int i = 0; i < mesh.nverts(); ++i) {
            optimizer.set_vertex_dim_lock(i, 2);
        }        
    }
    else {
        for (unsigned i = mesh.nverts(); i < mesh_wrapper.nb_vertices(); ++i) {
            optimizer.set_vertex_Lock(i);
        }
    }

    optimizer.set_verbose();


    if (is_2d_case) {
        optimizer.run();
        mesh_wrapper.get_scaffold_meshes(scaffold_tet, scaffold_pyramids);
        UM::write_by_extension(identifier + "_output.mesh", mesh);
        UM::write_by_extension(identifier + "_output_scaffold_t.mesh", scaffold_tet);
        UM::write_by_extension(identifier + "_output_scaffold_p.mesh", scaffold_pyramids);
        UM::write_by_extension(identifier + "_output_curves.mesh", boundary_loop);

        return EXIT_SUCCESS;
    }

    for (unsigned i = 1; i < 6; ++i) {
        mesh_wrapper.update_scaffold();
        optimizer.run();
        mesh_wrapper.get_scaffold_meshes(scaffold_tet, scaffold_pyramids);

        UM::write_by_extension(identifier + "_iter_"+std::to_string(i)+"_scaffold_t.mesh", scaffold_tet);
        UM::write_by_extension(identifier + "_iter_"+std::to_string(i)+"_scaffold_p.mesh", scaffold_pyramids);
        UM::write_by_extension(identifier + "_iter_"+std::to_string(i)+".mesh", mesh);
    } 
    UM::write_by_extension(identifier + "_output.mesh", mesh);
    if (is_open_mesh) UM::write_by_extension(identifier + "_output_curves.mesh", boundary_loop);
    

    return EXIT_SUCCESS;
}
