/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Inclusion interaction manager
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_INCLUSION_H
#define EXADIS_INCLUSION_H

#include "types.h"
#include <cmath>
#include <vector>
#include <unordered_map>

namespace ExaDiS {

class System;

class InclusionManager {
public:
    bool enabled = false;
    double a_dim = 0.0;
    std::vector<Vec3> centers;
    bool centers_valid = false;
    bool initialized = false;
    int orowan_loop_count = 0;
    std::unordered_map<long long, bool> node_was_inside;

    void initialize(System* system, SerialDisNet* network);

    Vec3 surface_normal(const Vec3& pos) const {
        double half = a_dim * 0.5;
        double min_dist = 1e30;
        Vec3 normal(0.0, 0.0, 1.0);
        double tol = a_dim * 0.05;

        for (const auto& center : centers) {
            Vec3 d = pos - center;
            double dx = fabs(fabs(d.x) - half);
            double dy = fabs(fabs(d.y) - half);
            double dz = fabs(fabs(d.z) - half);
            double dist = fmin(dx, fmin(dy, dz));

            if (dist < min_dist) {
                min_dist = dist;
                Vec3 nx(d.x > 0 ? 1.0 : -1.0, 0.0, 0.0);
                Vec3 ny(0.0, d.y > 0 ? 1.0 : -1.0, 0.0);
                Vec3 nz(0.0, 0.0, d.z > 0 ? 1.0 : -1.0);

                bool on_x = (dx < tol);
                bool on_y = (dy < tol);
                bool on_z = (dz < tol);

                Vec3 n(0.0);
                if (on_x) n += nx;
                if (on_y) n += ny;
                if (on_z) n += nz;

                double nmag = n.norm();
                if (nmag > 1e-10)
                    normal = (1.0/nmag) * n;
                else if (dx < dy && dx < dz)
                    normal = nx;
                else if (dy < dx && dy < dz)
                    normal = ny;
                else
                    normal = nz;
            }
        }
        return normal;
    }

    bool on_edge(const Vec3& pos, double edge_tol) const {
        if (centers.empty()) return false;
        int best = 0; double bestd = 1e30;
        for (int k = 0; k < (int)centers.size(); k++) {
            Vec3 d = pos - centers[k];
            double dd = dot(d, d);
            if (dd < bestd) { bestd = dd; best = k; }
        }
        Vec3 l = pos - centers[best];
        double half = a_dim * 0.5;
        int near = (fabs(l.x) >= half-edge_tol) + (fabs(l.y) >= half-edge_tol)
                 + (fabs(l.z) >= half-edge_tol);
        return near >= 2;
    }

    int single_face_normal(const Vec3& pos, Vec3& normal) const {
        normal = Vec3(0.0);
        if (centers.empty()) return -1;
        int best = 0; double bestd = 1e30;
        for (int k = 0; k < (int)centers.size(); k++) {
            Vec3 d = pos - centers[k];
            double dd = dot(d, d);
            if (dd < bestd) { bestd = dd; best = k; }
        }
        Vec3 local = pos - centers[best];
        double ax = fabs(local.x), ay = fabs(local.y), az = fabs(local.z);
        if (ax >= ay && ax >= az)
            normal = Vec3(local.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0);
        else if (ay >= ax && ay >= az)
            normal = Vec3(0.0, local.y >= 0.0 ? 1.0 : -1.0, 0.0);
        else
            normal = Vec3(0.0, 0.0, local.z >= 0.0 ? 1.0 : -1.0);
        return best;
    }

    bool is_node_in_inclusion(const Vec3& pos) const;
    bool is_node_strictly_inside_inclusion(const Vec3& pos) const;

    void before_integrate(System* system, SerialDisNet* network);
    void after_integrate(System* system, SerialDisNet* network);
    void after_reset_glide(System* system, SerialDisNet* network);
    void after_collision(System* system, SerialDisNet* network);
    void after_topology_remesh(System* system, SerialDisNet* network);

    void update_constraints(System* system, SerialDisNet* network);
    void project_surface_node_velocity(System* system, SerialDisNet* network);
    void insert_surface_nodes(System* system, SerialDisNet* network);
    void correct_surface_node_positions(System* system, SerialDisNet* network);
    void insert_edge_nodes(System* system, SerialDisNet* network);
    void detect_orowan_loop(SerialDisNet* network);

private:
    bool seg_cube_intersect(const Vec3& p_out, const Vec3& p_in,
                            const Vec3& center, double half, Vec3& hit) const;
    bool seg_cube_clip(const Vec3& pa, const Vec3& pb,
                       const Vec3& center, double half,
                       Vec3& hit_in, Vec3& hit_out) const;
};

} // namespace ExaDiS

#endif
