#pragma once
#include <set>
#include <map>
#include <ultimaille/all.h>
#include <Mesh_optimization/mesh_representations.h>
#include "ultimaille_mesh_utils.h"


namespace UM_extension {
    using Mesh_optimization::utils::Contiguous_unsigned_range;
    using Mesh_optimization::helper_structures::Mixed_element_mesh;




    Eigen::Vector3d ultimaille2eigen(UM::vec3 v) {
        return {v[0], v[1], v[2]};
    }

    UM::vec3 eigen2ultimaille(Eigen::Vector3d v) {
        return {v[0], v[1], v[2]};
    }

    template <typename VolumeMeshType, typename SurfaceMeshType>
    void extract_boundary(VolumeMeshType &volume, SurfaceMeshType &surface, std::vector<bool> *vertex_on_boundary = nullptr) {
        volume.connect();
        clean_surface(surface);
        surface.points.data->assign(volume.points.begin(), volume.points.end());
        if (vertex_on_boundary != nullptr)
            vertex_on_boundary->resize(volume.nverts(), false);
        for (auto f : volume.iter_facets()) {
            if  (f.on_boundary()) {
                for (int j=0; j < static_cast<int>(f.size()); ++j) {
                    surface.facets.push_back(f.vertex(j));
                    if (vertex_on_boundary != nullptr)
                        (*vertex_on_boundary)[f.vertex(j)] = true;
                }
            }
        }
    }


    class Tetrahedral_mesh_wrapper {
    public:
        using Cell_descriptor = int;
        using Vertex_descriptor = int;
        using Point_3 = UM::vec3;

        unsigned nb_cells() const { return tetmesh.ncells(); }
        unsigned nb_vertices() const { return tetmesh.nverts(); }

        Point_3 vertex_coordinates(Vertex_descriptor vertex) const { return tetmesh.points[vertex]; }
        void set_new_vertex_coordinates(int vertex, UM::vec3 coord) { tetmesh.points[vertex] = coord; }

        Contiguous_unsigned_range cell_range() const {
            return Contiguous_unsigned_range{0, nb_cells()};
        }
        Vertex_descriptor const * cell_vertices(Cell_descriptor cell) const { return &(tetmesh.cells.data())[4*cell]; }
        std::array<Point_3, 4> cell_reference_shape(Cell_descriptor cell) const { return refmesh == nullptr ? ref_shape : get_cell_ref(cell); }
    public:
        Tetrahedral_mesh_wrapper(UM::Tetrahedra &mesh, bool inverse_numbering = false, UM::Tetrahedra const *ref = nullptr)
        : tetmesh(mesh)
        , refmesh(ref)
        {
            ref_shape.inverse = inverse_numbering;
        }
        UM::Tetrahedra &tetmesh;
        UM::Tetrahedra const *refmesh;
        Mesh_optimization::Shapes::VTK_TETRAHEDRON<UM::vec3> ref_shape;

        std::array<Point_3, 4> get_cell_ref(Cell_descriptor cell) const {
            return {refmesh->points[refmesh->vert(cell, 0)], refmesh->points[refmesh->vert(cell, 1)], refmesh->points[refmesh->vert(cell, 2)], refmesh->points[refmesh->vert(cell, 3)]};
        }
    };

    class Triangle_boundary_wrapper {
    public:
        using Face_descriptor = int;
        using Normal_3 = UM::vec3;
        using Surface_patch_index = unsigned;
        unsigned nb_faces() const { return trimesh.nfacets(); }
        Contiguous_unsigned_range face_range() const {
            return Contiguous_unsigned_range{0, nb_faces()};
        }
        unsigned patch_id(Face_descriptor f) const { return f; }
        unsigned nb_face_vertices(Face_descriptor face) const { return 3; }
        auto face_vertices(Face_descriptor face) const { return std::array<int, 3>{trimesh.vert(face, 0), trimesh.vert(face, 1), trimesh.vert(face, 2)}; }

    public:
        UM::Triangles &trimesh;
    };

    class PolyLine_wrapper {
    public:
        using Edge_descriptor = int;
        using Curve_index = unsigned;
        unsigned nb_edges() const { return segments.nedges(); }
        Contiguous_unsigned_range edge_range() const {
            return Contiguous_unsigned_range{0, nb_edges()};
        }
        unsigned curve_id(Edge_descriptor e) const { return e; }
        int edge_vertex(Edge_descriptor edge, unsigned i) const { return segments.vert(edge, i); }
    public:
        UM::PolyLine &segments;
    };

    class Hexahedral_mesh_wrapper {
    public:
        using Cell_descriptor = int;
        using Vertex_descriptor = int;
        using Point_3 = UM::vec3;

        unsigned nb_cells() const { return static_cast<unsigned>(8*hexmesh.ncells()); }
        unsigned nb_vertices() const { return static_cast<unsigned>(hexmesh.nverts()); }

        Point_3 vertex_coordinates(Vertex_descriptor vertex) const { return hexmesh.points[vertex]; }
        void set_new_vertex_coordinates(int vertex, UM::vec3 coord) { hexmesh.points[vertex] = coord; }

        Contiguous_unsigned_range cell_range() const {
            return Contiguous_unsigned_range{0, nb_cells()};
        }
        std::array<Vertex_descriptor, 4> cell_vertices(Cell_descriptor cell) const {
            std::array<Vertex_descriptor, 4> tet_verts;
            for (unsigned i = 0; i < 4; ++i) {
                tet_verts[i] = hexmesh.vert(cell/8, hex_ref.inner_tetrahedra_local_vert(cell%8, i));
            }
            return tet_verts;
        }
        std::array<UM::vec3, 4> cell_reference_shape(Cell_descriptor cell) const { return hex_ref.inner_tetrahedra_reference_shape(cell%8); }
    public:
        Hexahedral_mesh_wrapper(UM::Hexahedra &mesh, bool inverse_numbering = false)
        : hexmesh(mesh)
        {
            hex_ref.inverse = inverse_numbering;
        }
        UM::Hexahedra &hexmesh;
        Mesh_optimization::Shapes::GEOGRAM_HEXAHEDRON<UM::vec3> hex_ref;
    };

    class Quad_boundary_wrapper {
    public:
        using Face_descriptor = int;
        using Normal_3 = UM::vec3;
        using Surface_patch_index = unsigned;
        unsigned nb_faces() const { return quadmesh.nfacets(); }
        Contiguous_unsigned_range face_range() const {
            return Contiguous_unsigned_range{0, nb_faces()};
        }
        unsigned patch_id(Face_descriptor f) const { return f; }
        unsigned nb_face_vertices(Face_descriptor face) const { return 4; }
        auto face_vertices(Face_descriptor face) const { return std::array<int, 4>{quadmesh.vert(face, 0), quadmesh.vert(face, 1), quadmesh.vert(face, 2), quadmesh.vert(face, 3)}; }

    public:
        UM::Quads &quadmesh;
    };


    class Mixed_mesh_container : public Mixed_element_mesh<int, int, UM::vec3, Contiguous_unsigned_range> {
    public:
        std::size_t nb_vertices() const override { return tm.nverts(); }
        UM::vec3 vertex_coordinates(int vertex) const override{return tm.points[vertex]; };
        void set_new_vertex_coordinates(int vertex, UM::vec3 coord) override {
            tm.points[vertex] = coord;
        }
        Contiguous_unsigned_range input_cell_range() const override{
            return Contiguous_unsigned_range { 0, static_cast<unsigned>(tm.ncells() + pm.ncells() + wm.ncells() + hm.ncells()) };
        }

        int get_cell_vertex(int cell_descriptor, unsigned local_Vertex_descriptor) const override {
            if (cell_descriptor < tm.ncells()) {
                return tm.vert(cell_descriptor, local_Vertex_descriptor);
            }
            cell_descriptor -= tm.ncells();
            if (cell_descriptor < pm.ncells()) {
                return pm.vert(cell_descriptor, local_Vertex_descriptor);

            }
            cell_descriptor -= pm.ncells();
            if (cell_descriptor < wm.ncells()) {
                return wm.vert(cell_descriptor, local_Vertex_descriptor);
            }
            cell_descriptor -= wm.ncells();
            return hm.vert(cell_descriptor, local_Vertex_descriptor);
        }


        Shape const * get_element_shape(int cell) const override {
            if (cell < tm.ncells()) {
                return &tet_ref;
            }
            cell -= tm.ncells();
            if (cell < pm.ncells()) {
                return &py_ref;

            }
            cell -= pm.ncells();
            if (cell < wm.ncells()) {
                return &we_ref;
            }
            cell -= wm.ncells();
            return &hex_ref;
        }

        void set_orientation(bool inv_tet, bool inv_hex, bool inv_pyr, bool inv_wed) {
            tet_ref.inverse = inv_tet;
            hex_ref.inverse = inv_hex;
            py_ref.inverse  = inv_pyr;
            we_ref.inverse  = inv_wed;
        }

        Mixed_mesh_container(
            UM::Tetrahedra &tm_,
            UM::Pyramids const &pm_,
            UM::Wedges const &wm_,
            UM::Hexahedra const &hm_
        ) : tm(tm_), pm(pm_), wm(wm_), hm(hm_) {
            this->assemble();
        }

        UM::Tetrahedra &tm;
        UM::Pyramids const &pm;
        UM::Wedges const &wm;
        UM::Hexahedra const &hm;

        Mesh_optimization::Shapes::VTK_TETRAHEDRON<UM::vec3> tet_ref;
        Mesh_optimization::Shapes::VTK_PYRAMID<UM::vec3> py_ref;
        Mesh_optimization::Shapes::VTK_WEDGE<UM::vec3> we_ref;
        Mesh_optimization::Shapes::GEOGRAM_HEXAHEDRON<UM::vec3> hex_ref;
    };

    using Polygonal_boundary_support = Mesh_optimization::helper_structures::Polygonal_boundary<unsigned, unsigned, UM::vec3>;


   inline void generate_boundary(
        UM::Tetrahedra &tm,
        UM::Pyramids &pm,
        UM::Wedges &wm,
        UM::Hexahedra &hm,
        Polygonal_boundary_support &boundary,
        std::vector<bool> &vert_on_boundary,
        UM::Triangles * associated_boundary_mesh = nullptr
    )
    {
        std::map<std::set<unsigned>, std::pair<unsigned, std::vector<unsigned>>> faceCnt;
        std::set<unsigned> tri;
        std::vector<unsigned> vectri;
        for (auto mesh: std::vector<UM::Volume *>{&tm, &pm, &wm, &hm}) {
            for (auto facet : mesh->iter_facets()) {
                if  (facet.on_boundary()) {
                    tri.clear();
                    vectri.clear();
                    for (int j =0; j < facet.size(); ++j) {
                        tri.insert(facet.vertex(j));
                        vectri.push_back(facet.vertex(j));
                    }
                    auto res = faceCnt.emplace(tri, std::pair<unsigned, std::vector<unsigned>>{0, vectri});
                    ++res.first->second.first; // oof
                }
            }
        }

        if (associated_boundary_mesh != nullptr) associated_boundary_mesh->points.data->assign(tm.points.begin(), tm.points.end());
        vert_on_boundary.clear();
        vert_on_boundary.resize(tm.nverts(), false);
        for (auto [set, pair] : faceCnt) {
            unsigned cnt = pair.first;
            auto const &vec = pair.second;
            if (cnt > 1) continue;
            boundary.add_polygon(vec);
            for (unsigned v : vec) {
                vert_on_boundary[v] = true;
            }
            if (associated_boundary_mesh == nullptr) continue;
            if (vec.size() == 3) {
                for (unsigned i = 0; i < 3; ++i) {
                    associated_boundary_mesh->facets.push_back(vec[i]);
                }
            }
            if (vec.size() == 4) {
                UM::vec3 center = 0.25*(tm.points[vec[0]] + tm.points[vec[1]] + tm.points[vec[2]] + tm.points[vec[3]]);
                int center_id = associated_boundary_mesh->points.push_back(center);
                for (unsigned i = 0; i < 4; ++i) {
                    associated_boundary_mesh->facets.push_back(vec[(i+0)%4]);
                    associated_boundary_mesh->facets.push_back(vec[(i+1)%4]);
                    associated_boundary_mesh->facets.push_back(center_id);
                }
            }

        }
    }

    struct Mixed_element_mesh_serializer {
        void load(std::string const &filename) {
            clean_mesh(tm);
            clean_mesh(pm);
            clean_mesh(wm);
            clean_mesh(hm);
            UM::read_by_extension(filename, tm);
            UM::read_by_extension(filename, pm);
            UM::read_by_extension(filename, wm);
            UM::read_by_extension(filename, hm);
            pm.points = tm.points;
            wm.points = tm.points;
            hm.points = tm.points;
            tm.connect();
            pm.connect();
            wm.connect();
            hm.connect();
            only_hex = (tm.ncells() == 0 && pm.ncells() == 0 && wm.ncells() == 0 && hm.ncells() > 0);
            only_tet = (tm.ncells() > 0 && pm.ncells() == 0 && wm.ncells() == 0 && hm.ncells() == 0);

            std::cout << "Loaded mixed mesh: " << std::endl;
            std::cout  << "#tets: " << tm.ncells() << std::endl;
            std::cout  << "#pyramids: " << pm.ncells() << std::endl;
            std::cout  << "#wedges: " << wm.ncells() << std::endl;
            std::cout  << "#hexes: " << hm.ncells() << std::endl;
            std::cout  << "#Vertices: " << hm.nverts() << std::endl;
            if (only_hex) {
                std::cout << "Mesh is only hexahedral." << std::endl;
            }
            if (only_tet) {
                std::cout << "Mesh is only tetrahedral." << std::endl;
            }

            on_boundary.resize(tm.nverts(), false);
            UM_extension::generate_boundary(tm, pm, wm, hm, poly_boundary, on_boundary, &triangulated_surface);

            mixed_mesh_ptr.reset(new Mixed_mesh_container(tm, pm, wm, hm));

        }

        void save(std::string const &filename, std::string const &extension=".vtk") const {
            if (only_hex) {
                std::cout << "Saving hex mesh: " << filename + extension << std::endl;
                UM::write_by_extension(filename + extension, hm);
            }
            else if (only_tet) {
                std::cout << "Saving tet mesh: " << filename + extension << std::endl;
                UM::write_by_extension(filename + extension, tm);
            }
            else {
                std::cout << "Saving mixed mesh: " << filename << "..." << extension << std::endl;
                UM::write_by_extension(filename + "_t" + extension, tm);
                UM::write_by_extension(filename + "_p" + extension, pm);
                UM::write_by_extension(filename + "_w" + extension, wm);
                UM::write_by_extension(filename + "_h" + extension, hm);
            }
        }

        UM::PointSet & get_points() {
            return tm.points;
        }

        UM::Triangles const & get_triangulated_surface() const {
            return triangulated_surface;
        }

        std::vector<bool> const & get_vertex_on_boundary() const {
            return on_boundary;
        }

        Mixed_mesh_container & get_mixed_mesh(
            bool inv_tet = false,
            bool inv_hex = false,
            bool inv_pyr = false,
            bool inv_wed = false 
        ) const {
            mixed_mesh_ptr->set_orientation(inv_tet, inv_hex, inv_pyr, inv_wed);
            return *mixed_mesh_ptr;
        }

        Polygonal_boundary_support const & get_polygonal_boundary() const {
            return poly_boundary;
        }

    public:
        bool only_hex = false;
        bool only_tet = false;

        UM::Tetrahedra tm;
        UM::Pyramids pm;
        UM::Wedges wm;
        UM::Hexahedra hm;

        UM::Triangles triangulated_surface;
        std::vector<bool> on_boundary;

        Polygonal_boundary_support poly_boundary;
        std::unique_ptr<Mixed_mesh_container> mixed_mesh_ptr;

    };



}

