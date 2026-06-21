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

// Project + clamp: pin constrained axes to ±half, clamp free axes to [-half,+half].
// Prevents node from drifting outside the face's square boundary (no-switch model).
KOKKOS_INLINE_FUNCTION
Vec3 inclusion_project_clamp(const Vec3& local, const int face_sign[3], double half) {
    Vec3 p = local;
    for (int k = 0; k < 3; k++) {
        if (face_sign[k] != 0) p[k] = face_sign[k] * half;
        else { if (p[k] >  half) p[k] =  half;
               if (p[k] < -half) p[k] = -half; }
    }
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

// Capture projection: pin face axes to ±half; for free axes that overshoot
// (|p|>half), pull them just inside to ±(half-margin). In-range free axes
// are untouched, so legitimate near-edge face nodes are unaffected.
KOKKOS_INLINE_FUNCTION
Vec3 inclusion_project_capture(const Vec3& local, const int face_sign[3], double half, double margin) {
    Vec3 p = local;
    for (int k = 0; k < 3; k++) {
        if (face_sign[k] != 0) p[k] = face_sign[k] * half;
        else { if (p[k] >  half) p[k] =  half - margin;
               if (p[k] < -half) p[k] = -half + margin; }
    }
    return p;
}

// Project local onto the 1D line (face ∩ glide-plane):
// pin the face axis to ±half, then slide within the face along glide_n's
// in-face component back onto the glide plane (minimal correction), and
// clamp the free axes to the box edges (node naturally anchors at an edge).
// glide_n = dislocation glide-plane normal (lab axes = inclusion-local axes,
// since the inclusion is an axis-aligned cube).
KOKKOS_INLINE_FUNCTION
Vec3 inclusion_project_glide_line(const Vec3& local, const int face_sign[3],
                                  double half, const Vec3& glide_n) {
    int a = (face_sign[0] != 0) ? 0 : ((face_sign[1] != 0) ? 1 : 2);
    int b = (a + 1) % 3, c = (a + 2) % 3;
    Vec3 p = local;
    double da = face_sign[a] * half - local[a];
    p[a] = face_sign[a] * half;
    double denom = glide_n[b]*glide_n[b] + glide_n[c]*glide_n[c];
    if (denom > 1e-12) {
        double t = -glide_n[a] * da / denom;
        p[b] = local[b] + t * glide_n[b];
        p[c] = local[c] + t * glide_n[c];
    }
    // 落点超出面方块 → 沿"面∩滑移面"这条线(方向 ⟂ (n[b],n[c]))退回方块边界,
    // 落在 π∩棱(保持在滑移面上 = 截线 P 的棱点),而非各轴独立 clamp 到盒角(脱滑移面、跑顶点)。
    if (p[b] > half || p[b] < -half || p[c] > half || p[c] < -half) {
        double db = -glide_n[c], dc = glide_n[b];   // in-face line direction (on π)
        if (db*db + dc*dc > 1e-12) {
            double s_lo = -1e30, s_hi = 1e30;
            if (db > 1e-12 || db < -1e-12) {
                double s1 = (-half - p[b]) / db, s2 = (half - p[b]) / db;
                if (s1 > s2) { double tmp = s1; s1 = s2; s2 = tmp; }
                if (s1 > s_lo) s_lo = s1;
                if (s2 < s_hi) s_hi = s2;
            }
            if (dc > 1e-12 || dc < -1e-12) {
                double s1 = (-half - p[c]) / dc, s2 = (half - p[c]) / dc;
                if (s1 > s2) { double tmp = s1; s1 = s2; s2 = tmp; }
                if (s1 > s_lo) s_lo = s1;
                if (s2 < s_hi) s_hi = s2;
            }
            if (s_lo <= s_hi) {                       // clip s=0(越界点)回 [s_lo,s_hi]
                double s = (0.0 < s_lo) ? s_lo : (0.0 > s_hi ? s_hi : 0.0);
                p[b] += s * db;
                p[c] += s * dc;
            }
        }
        if (p[b] >  half) p[b] =  half;   // 浮点兜底
        if (p[b] < -half) p[b] = -half;
        if (p[c] >  half) p[c] =  half;
        if (p[c] < -half) p[c] = -half;
    }
    return p;
}

// inclusion_segment_edge_cross / inclusion_project_edge_point 已删除(block-not-stick):
// 不再插棱角点、不再钉死边角节点(节点到棱即释放为自由节点),两个 helper 已无调用者。

} // namespace ExaDiS

#endif
