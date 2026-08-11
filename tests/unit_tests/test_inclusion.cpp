/*---------------------------------------------------------------------------
 *
 *  Focused regression tests for the axis-aligned cubic inclusion model.
 *
 *-------------------------------------------------------------------------*/

#include <cmath>
#include <iostream>
#include <string>

#include "system.h"
#include "inclusion_geometry.h"

using namespace ExaDiS;

namespace {

int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        failures++;
    }
}

InclusionManager make_cube()
{
    InclusionManager inclusion;
    inclusion.enabled = true;
    inclusion.initialized = true;
    inclusion.a_dim = 20.0;
    inclusion.centers.push_back(Vec3(50.0, 50.0, 50.0));
    inclusion.centers_valid = true;
    return inclusion;
}

void add_single_segment(SerialDisNet& network, const Vec3& p1, const Vec3& p2)
{
    network.add_node(p1);
    network.add_node(p2);
    network.add_seg(0, 1, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0));
    network.generate_connectivity();
    network.update_ptr();
}

void test_surface_classification()
{
    InclusionManager inclusion = make_cube();

    Vec3 normal;
    check(inclusion.single_face_normal(Vec3(60.0, 50.0, 50.0), normal) == 0,
          "face point must be assigned to the cube");
    check((normal-Vec3(1.0, 0.0, 0.0)).norm2() < 1e-12,
          "positive-x face normal must be +x");
    check(!inclusion.on_edge(Vec3(60.0, 50.0, 50.0), 1e-6),
          "single-face point must not be classified as an edge");
    check(inclusion.on_edge(Vec3(60.0, 60.0, 50.0), 1e-6),
          "two-face point must be classified as an edge");
    check(inclusion.on_edge(Vec3(60.0, 60.0, 60.0), 1e-6),
          "three-face point must be classified as an edge/corner");

    int face_sign[3];
    inclusion_classify(Vec3(10.0, 0.0, 0.0), 10.0, 1e-8, face_sign);
    check(inclusion_face_count(face_sign) == 1,
          "geometry helper must classify a face point as one touched face");
    inclusion_classify(Vec3(10.0, 10.0, 0.0), 10.0, 1e-8, face_sign);
    check(inclusion_face_count(face_sign) == 2,
          "geometry helper must classify an edge point as two touched faces");
    inclusion_classify(Vec3(10.0, 10.0, 10.0), 10.0, 1e-8, face_sign);
    check(inclusion_face_count(face_sign) == 3,
          "geometry helper must classify a corner point as three touched faces");
}

void test_non_intersecting_segment()
{
    InclusionManager inclusion = make_cube();
    SerialDisNet network(100.0);
    add_single_segment(network, Vec3(10.0, 10.0, 50.0),
                       Vec3(20.0, 20.0, 50.0));
    inclusion.insert_surface_nodes(nullptr, &network);
    check(network.number_of_nodes() == 2,
          "non-intersecting segment must not gain nodes");
    check(network.number_of_segs() == 1,
          "non-intersecting segment must not be split");
}

void test_outside_outside_adjacent_face_clip()
{
    InclusionManager inclusion = make_cube();
    // The segment enters through x=-half and exits through y=+half.
    // Its z=constant glide plane intersects the shared cube edge.
    SerialDisNet network(100.0);
    add_single_segment(network, Vec3(35.0, 50.0, 50.0),
                       Vec3(50.0, 65.0, 50.0));
    inclusion.insert_surface_nodes(nullptr, &network);

    int inclusion_nodes = 0;
    for (const auto& node : network.nodes)
        if (node.constraint == INCLUSION_NODE) inclusion_nodes++;

    check(network.number_of_nodes() == 5,
          "outside-outside adjacent-face clip must add entry, exit and edge nodes");
    check(network.number_of_segs() == 4,
          "outside-outside adjacent-face clip must be routed as four segments");
    check(inclusion_nodes == 3,
          "entry, exit and edge nodes must carry INCLUSION_NODE constraints");

    for (int i = 0; i < network.number_of_segs(); i++) {
        if (network.segs[i].burg.norm2() < 1e-20) continue;
        Vec3 p1 = network.nodes[network.segs[i].n1].pos;
        Vec3 p2 = network.cell.pbc_position(p1, network.nodes[network.segs[i].n2].pos);
        Vec3 midpoint = 0.5 * (p1+p2);
        check(!inclusion.is_node_strictly_inside_inclusion(midpoint),
              "routed non-zero segment midpoint must not lie inside the cube");
    }
}

void test_surface_velocity_projection()
{
    InclusionManager inclusion = make_cube();
    SerialDisNet network(100.0);
    network.add_node(Vec3(60.0, 50.0, 50.0), INCLUSION_NODE);
    network.add_node(Vec3(60.0, 55.0, 50.0));
    network.add_node(Vec3(60.0, 60.0, 50.0), INCLUSION_NODE);
    network.add_seg(0, 1, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0));
    network.generate_connectivity();
    network.update_ptr();
    network.nodes[0].v = Vec3(1.0, 2.0, 3.0);
    network.nodes[2].v = Vec3(1.0, 2.0, 3.0);

    inclusion.project_surface_node_velocity(nullptr, &network);

    check((network.nodes[0].v-Vec3(0.0, 2.0, 0.0)).norm2() < 1e-12,
          "face-node velocity must follow the surface/glide-plane intersection");
    check(network.nodes[2].v.norm2() < 1e-12,
          "edge-node velocity must be zero");
}

void test_bypass_candidate_deduplication()
{
    InclusionManager inclusion = make_cube();
    SerialDisNet network(100.0);
    network.add_node(Vec3(40.0, 50.0, 50.0), INCLUSION_NODE);
    network.add_node(Vec3(60.0, 50.0, 50.0), INCLUSION_NODE);
    network.add_seg(0, 1, Vec3(0.0), Vec3(0.0, 0.0, 1.0));
    network.generate_connectivity();
    network.update_ptr();

    inclusion.detect_orowan_loop(&network);
    inclusion.detect_orowan_loop(&network);
    check(inclusion.bypass_candidate_count == 1,
          "the same active ghost chord must not be counted twice");
    check(inclusion.orowan_loop_count == 0,
          "a ghost chord alone must not increment the Orowan-loop count");

    network.segs[0].burg = Vec3(1.0, 0.0, 0.0);
    inclusion.detect_orowan_loop(&network);
    network.segs[0].burg = Vec3(0.0);
    inclusion.detect_orowan_loop(&network);
    check(inclusion.bypass_candidate_count == 2,
          "a candidate that disappears and later reappears must be a new event");
}

} // namespace

int main(int argc, char* argv[])
{
    ExaDiS::Initialize init(argc, argv);
    test_surface_classification();
    test_non_intersecting_segment();
    test_outside_outside_adjacent_face_clip();
    test_surface_velocity_projection();
    test_bypass_candidate_deduplication();

    if (failures > 0) {
        std::cerr << failures << " inclusion test(s) failed\n";
        return 1;
    }
    return 0;
}
