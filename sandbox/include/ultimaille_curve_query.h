#pragma once
#include <ultimaille/helpers/hboxes.h>
#include <ultimaille/surface.h>

#include "ultimaille_distances.h"

namespace UM_extension {

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
        UM::PolyLine::Edge best_edge = {m, (int) 0};
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

}
