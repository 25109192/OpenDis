/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    Unified inclusion surface geometry helpers (refactor stage 2).
 *    Pure functions, device-callable, no map access.
 *
 *    local = pos - inclusion_center
 *    half  = inclusion_a_dim * 0.5
 *    tol   = face tolerance (e.g. 100 b)
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_INCLUSION_GEOMETRY_H
#define EXADIS_INCLUSION_GEOMETRY_H

#include "vec.h"

namespace ExaDiS {

// Classify which faces a node touches.
// face_sign[k]: +1 = near +half face, -1 = near -half face, 0 = free axis.
// Criterion: |local[k] ∓ half| <= tol  (upper-bounded, only recognises near ±half)
KOKKOS_INLINE_FUNCTION
void inclusion_classify(const Vec3& local, double half, double tol, int face_sign[3]) {
    for (int k = 0; k < 3; k++) {
        if      (fabs(local[k] - half) <= tol) face_sign[k] = +1;
        else if (fabs(local[k] + half) <= tol) face_sign[k] = -1;
        else                                    face_sign[k] =  0;
    }
}

// Number of faces the node touches: 1=face, 2=edge, 3=corner.
KOKKOS_INLINE_FUNCTION
int inclusion_face_count(const int face_sign[3]) {
    return (face_sign[0] != 0) + (face_sign[1] != 0) + (face_sign[2] != 0);
}

// Outward unit normal from face_sign (sum of touched-face normals, normalised).
KOKKOS_INLINE_FUNCTION
Vec3 inclusion_normal(const int face_sign[3]) {
    Vec3 n(0.0);
    for (int k = 0; k < 3; k++) n[k] = (double)face_sign[k];
    double m = n.norm();
    if (m > 1e-10) n = (1.0/m) * n;
    return n;
}

// Project local onto the face/edge/corner described by face_sign:
// pin the constrained axes to ±half, leave free axes unchanged.
KOKKOS_INLINE_FUNCTION
Vec3 inclusion_project(const Vec3& local, const int face_sign[3], double half) {
    Vec3 p = local;
    for (int k = 0; k < 3; k++)
        if (face_sign[k] != 0) p[k] = face_sign[k] * half;
    return p;
}

// Force single-face assignment (for FLOW nodes): find the axis where
// |local[k]| is largest and assign that as the only face.
KOKKOS_INLINE_FUNCTION
void inclusion_nearest_face(const Vec3& local, int face_sign[3]) {
    face_sign[0] = face_sign[1] = face_sign[2] = 0;
    int best = 0;
    double bestd = fabs(local[0]);
    for (int k = 1; k < 3; k++) {
        double d = fabs(local[k]);
        if (d > bestd) { bestd = d; best = k; }
    }
    face_sign[best] = (local[best] >= 0.0) ? +1 : -1;
}

} // namespace ExaDiS

#endif
