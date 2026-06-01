/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Geometry utilities for inclusion handling
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_GEOMETRY_UTILS_H
#define EXADIS_GEOMETRY_UTILS_H

#include "vec.h"

namespace ExaDiS {

// 判断点是否在立方体内（含边界）
inline bool point_in_cube(const Vec3& p, const Vec3& center, double half) {
    return (fabs(p.x - center.x) <= half &&
            fabs(p.y - center.y) <= half &&
            fabs(p.z - center.z) <= half);
}

// 将点投影到立方体表面上（最近点）
inline Vec3 closest_point_on_cube_surface(const Vec3& p, const Vec3& center, double half) {
    Vec3 rel = p - center;
    // 先限制在立方体内部
    double x = fmax(-half, fmin(half, rel.x));
    double y = fmax(-half, fmin(half, rel.y));
    double z = fmax(-half, fmin(half, rel.z));
    // 再推到最近的表面（三个方向中偏离最大的）
    double dx = fabs(rel.x - x);
    double dy = fabs(rel.y - y);
    double dz = fabs(rel.z - z);
    if (dx >= dy && dx >= dz) {
        x = (rel.x > 0) ? half : -half;
    } else if (dy >= dz) {
        y = (rel.y > 0) ? half : -half;
    } else {
        z = (rel.z > 0) ? half : -half;
    }
    return center + Vec3(x, y, z);
}
// 判断交点 hit 在立方体的哪个面上，返回该面的外法向量
// 结果只有六种可能：±x, ±y, ±z 方向的单位向量
inline Vec3 get_face_normal(const Vec3& hit, const Vec3& center, double half) {
    Vec3 local = hit - center;
    // 找哪个轴最接近 ±half（即离哪个面最近）
    int best_axis = 0;
    double best_dist = 1e30;
    for (int k = 0; k < 3; ++k) {
        double dist = fabs(fabs(local[k]) - half);
        if (dist < best_dist) {
            best_dist = dist;
            best_axis = k;
        }
    }
    Vec3 normal(0.0);
    normal[best_axis] = (local[best_axis] > 0.0) ? 1.0 : -1.0;
    return normal;
}

// 把速度向量投影到面上（去掉法向分量）
inline Vec3 project_velocity_to_face(const Vec3& v, const Vec3& normal) {
    return v - dot(v, normal) * normal;
}

// 判断表面节点是否已经离开了当前面的范围
// 即：节点在该面切向方向上是否超出了 ±half 的边界
inline bool node_left_face(const Vec3& pos, const Vec3& center, double half,
                            const Vec3& normal) {
    Vec3 local = pos - center;
    // 遍历三个轴，检查不是法向的那两个轴（切向）
    for (int k = 0; k < 3; ++k) {
        if (fabs(normal[k]) > 0.5) continue;  // 跳过法向轴
        if (fabs(local[k]) > half + 1e-6 * half) return true;
    }
    return false;
}
} // namespace ExaDiS

#endif