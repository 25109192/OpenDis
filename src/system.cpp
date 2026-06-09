/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "system.h"
#include "inclusion_geometry.h"

namespace ExaDiS {

FILE* flog = nullptr;

/*---------------------------------------------------------------------------
 *
 *    Function:     System::System()
 *
 *-------------------------------------------------------------------------*/
System::System()
{
#ifdef MPI
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
#else
    num_ranks = 1;
    proc_rank = 0;
#endif
    net_mngr = nullptr;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::initialize()
 *
 *-------------------------------------------------------------------------*/
void System::initialize(Params _params, Crystal _crystal, SerialDisNet *network)
{
    // 读取夹杂参数
    double inclusion_a_phys = 0.0;
    const char* a_str = std::getenv("INCLUSION_A");
    if (a_str != nullptr) {
        inclusion_a_phys = std::stod(a_str);
        if (inclusion_a_phys > 0.0) inclusion_enabled = true;
    }

    double vol_frac = 0.10;
    const char* vol_str = std::getenv("INCLUSION_VOL_FRAC");
    if (vol_str != nullptr) {
        vol_frac = std::stod(vol_str);
        if (vol_frac <= 0.0 || vol_frac > 1.0) {
            ExaDiS_log("Warning: INCLUSION_VOL_FRAC=%f out of range, using 0.10\n", vol_frac);
            vol_frac = 0.10;
        }
    }

    params = _params;
    crystal = _crystal;

#ifdef MPI
    ExaDiS_fatal("System::initialize not implemented for MPI\n");
#endif

    if (!network)
        ExaDiS_fatal("Error: undefined initial dislocation configuration\n");

    net_mngr = make_network_manager(network);

    // 初始化夹杂阵列（只初始化一次）
    if (inclusion_enabled && !inclusion_initialized) {

        inclusion_a_dim = inclusion_a_phys / params.burgmag;

        double Lx_dim = network->cell.H.xx();
        double Ly_dim = network->cell.H.yy();
        double Lz_dim = network->cell.H.zz();

        // 根据夹杂边长和体积分数自适应计算阵列数量
        double box_vol = Lx_dim * Ly_dim * Lz_dim;
        double inclusion_vol = inclusion_a_dim * inclusion_a_dim * inclusion_a_dim;
        int N_total = (int)round(vol_frac * box_vol / inclusion_vol);
        int Nx = (int)round(pow((double)N_total, 1.0/3.0));
        if (Nx < 1) Nx = 1;
        int Ny = Nx, Nz = Nx;

        double dx = Lx_dim / Nx;
        double dy = Ly_dim / Ny;
        double dz = Lz_dim / Nz;

        if (inclusion_a_dim > dx || inclusion_a_dim > dy || inclusion_a_dim > dz) {
            ExaDiS_log("Warning: inclusion size may cause overlap! a_dim=%.2f, dx=%.2f, dy=%.2f, dz=%.2f\n",
                       inclusion_a_dim, dx, dy, dz);
        }

        inclusion_centers.clear();
        for (int i = 0; i < Nx; ++i) {
            double cx = dx * (i + 0.5);
            for (int j = 0; j < Ny; ++j) {
                double cy = dy * (j + 0.5);
                for (int k = 0; k < Nz; ++k) {
                    double cz = dz * (k + 0.5);
                    inclusion_centers.push_back(Vec3(cx, cy, cz));
                }
            }
        }
        inclusion_centers_valid = true;

        ExaDiS_log("Inclusion array: %d x %d x %d = %d inclusions, a_dim=%.2f, vol_frac=%.2f\n",
                   Nx, Ny, Nz, (int)inclusion_centers.size(), inclusion_a_dim, vol_frac);
        for (size_t idx = 0; idx < std::min(10, (int)inclusion_centers.size()); ++idx) {
            ExaDiS_log("  Inclusion %zu: center=(%.2f, %.2f, %.2f)\n",
                       idx, inclusion_centers[idx].x, inclusion_centers[idx].y, inclusion_centers[idx].z);
        }

        inclusion_initialized = true;
    }

    if (inclusion_enabled) network->recycle = false;  // [VERIFY] disable index recycling to confirm stale-label inheritance root cause

    reset_glide_planes();
    update_inclusion_constraints(network);

    neighbor_cutoff = 0.0;
    extstress.zero();
    dEp.zero();
    dWp.zero();
    realdt = 0.0;
    density = network->dislocation_density(params.burgmag);

    oprec = new OpRec();
    network->oprec = oprec;

    for (int i = 0; i < TIMER_END; i++)
        timer[i].accumtime = 0.0;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::~System()
 *
 *-------------------------------------------------------------------------*/
System::~System()
{
    exadis_delete(net_mngr);
    if (oprec) delete oprec;
    if (flog) fclose(flog);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::register_neighbor_cutoff()
 *
 *-------------------------------------------------------------------------*/
void System::register_neighbor_cutoff(double cutoff)
{
    if (cutoff > neighbor_cutoff)
        neighbor_cutoff = cutoff;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::plastic_strain()
 *
 *-------------------------------------------------------------------------*/
template <class N>
class PlasticStrain {
private:
    System* system;
    N* net;
    double vol, bmag;
public:
    PlasticStrain(System* _system, N* _net) : system(_system), net(_net)
    {
        vol = net->cell.volume();
        bmag = system->params.burgmag;
        system->dEp.zero();
        system->dWp.zero();
        system->density = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (const team_handle& team) const
    {
        int ts = team.team_size();
        int lid = team.league_rank();

        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        auto cell = net->cell;

        Mat33 E, W;
        double rho;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, ts),
        [=](int& t, Mat33& Esum, Mat33& Wsum, double& rhosum) {
            int i = lid*ts + t;
            if (i < net->Nsegs_local) {
                int n1 = segs[i].n1;
                int n2 = segs[i].n2;
                Vec3 b = segs[i].burg;

                Vec3 r1 = nodes[n1].pos;
                Vec3 r2 = cell.pbc_position(r1, nodes[n2].pos);
                Vec3 r3 = cell.pbc_position(r1, system->xold(n1));
                Vec3 r4 = cell.pbc_position(r3, system->xold(n2));
                Vec3 n = 0.5*cross(r2-r3, r1-r4);

                Mat33 P = 1.0/vol * outer(n, b);
                Esum += 0.5 * (P + P.transpose());
                Wsum += 0.5 * (P - P.transpose());
                rhosum += 1.0/vol/bmag/bmag * (r2-r1).norm();
            }
        }, E, W, rho);

        Kokkos::single(Kokkos::PerTeam(team), [=]() {
            Kokkos::atomic_add(&system->dEp, E);
            Kokkos::atomic_add(&system->dWp, W);
            Kokkos::atomic_add(&system->density, rho);
        });
    }
};

/*---------------------------------------------------------------------------
 *
 *    Function:     System::is_node_in_inclusion()
 *
 *-------------------------------------------------------------------------*/
bool System::is_node_in_inclusion(const Vec3& pos) const {
    if (!inclusion_enabled) return false;
    double half = inclusion_a_dim * 0.5;
    double tol = 1e-6 * inclusion_a_dim;
    for (const auto& center : inclusion_centers) {
        if (pos.x >= center.x - half + tol && pos.x <= center.x + half - tol &&
            pos.y >= center.y - half + tol && pos.y <= center.y + half - tol &&
            pos.z >= center.z - half + tol && pos.z <= center.z + half - tol) {
            return true;
        }
    }
    return false;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::is_node_strictly_inside_inclusion()
 *
 *    【修复2】容差从 1e-3*a_dim 改为 0.05*a_dim
 *    确保表面 PINNED 节点（坐标误差约 1~10b）不被误判为内部节点。
 *    0.05*a_dim = 114b，远大于坐标误差，远小于夹杂半径（1137b）。
 *
 *-------------------------------------------------------------------------*/
bool System::is_node_strictly_inside_inclusion(const Vec3& pos) const {
    if (!inclusion_enabled) return false;
    double half = inclusion_a_dim * 0.5;
    // 容差从 1e-3 改为 0.05：表面节点坐标在 ±half 附近，
    // 内缩 0.05*a_dim 后表面节点不会被判为内部节点
    double tol = 0.05 * inclusion_a_dim;
    for (const auto& center : inclusion_centers) {
        if (pos.x > center.x - half + tol &&
            pos.x < center.x + half - tol &&
            pos.y > center.y - half + tol &&
            pos.y < center.y + half - tol &&
            pos.z > center.z - half + tol &&
            pos.z < center.z + half - tol) {
            return true;
        }
    }
    return false;
}

void System::plastic_strain()
{
    DeviceDisNet* net = get_device_network();
    int oldsz = (int)xold.extent(0);
    if (oldsz < net->Nnodes_local) {
        Kokkos::resize(xold, net->Nnodes_local);
        T_x xo = xold;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(oldsz, net->Nnodes_local),
            KOKKOS_LAMBDA(const int& i){
                auto nodes = net->get_nodes();
                xo(i) = nodes[i].pos;
            });
        Kokkos::fence();
    }
    TeamSize ts = get_team_sizes(net->Nsegs_local);
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(ts.num_teams, ts.team_size),
        PlasticStrain<DeviceDisNet>(this, net)
    );
    Kokkos::fence();

    if (oprec)
        oprec->add_op(OpRec::PlasticStrain(dEp, dWp, density));
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::update_inclusion_constraints()
 *
 *    第二阶段 2C 版本：
 *    扩展第一步的容差判据，能捕获：
 *    1. 严格在表面上的节点（含浮点级容差 TOL_INSIDE）
 *       —— Remesh bisect 中点的典型情形
 *    2. 表面外极近距离的节点（容差 TOL_OUTSIDE = 50 b）
 *       —— Collision 合并漂移的典型情形
 *
 *    所有被捕获的节点都会被投影到最近的夹杂面、PIN、登记。
 *
 *-------------------------------------------------------------------------*/
void System::update_inclusion_constraints(SerialDisNet* network) {
    if (!inclusion_enabled) return;
    if (network == nullptr) return;

    int nnodes = network->number_of_nodes();
    static int total_fixed_count = 0;
    int new_fixed = 0;
    int new_projected = 0;
    int new_outside = 0;
    double half = inclusion_a_dim * 0.5;

    const double TOL_INSIDE  = 1e-9 * inclusion_a_dim;  // 浮点级容差
    const double TOL_OUTSIDE = 50.0;                      // 单位 b，外部容差

    // ============================================================
    // 第一步：扫描节点，捕获在内部、表面上、或表面外极近的节点
    // ============================================================
    for (int i = 0; i < nnodes; ++i) {
        if (network->nodes[i].constraint == INCLUSION_NODE ||
            network->nodes[i].constraint == PINNED_NODE) continue;

        Vec3 pos = network->nodes[i].pos;

        // ---- 判定节点应该被处理的具体夹杂和情形 ----
        int  detected_incl  = -1;
        bool is_inside_or_on = false;
        bool is_just_outside = false;

        // 第一遍：查找"在内或在表面"的夹杂
        for (int k = 0; k < (int)inclusion_centers.size(); k++) {
            Vec3 local = pos - inclusion_centers[k];
            if (fabs(local.x) <= half + TOL_INSIDE &&
                fabs(local.y) <= half + TOL_INSIDE &&
                fabs(local.z) <= half + TOL_INSIDE) {
                is_inside_or_on = true;
                detected_incl = k;
                break;
            }
        }

        // 第二遍：如果不在内/表面，查找"表面外极近"的情形
        if (!is_inside_or_on) {
            double near_dist = 1e30;
            for (int k = 0; k < (int)inclusion_centers.size(); k++) {
                Vec3 local = pos - inclusion_centers[k];
                // 节点必须横向与某个夹杂"对齐"（在扩展范围内）
                if (fabs(local.x) > half + TOL_OUTSIDE) continue;
                if (fabs(local.y) > half + TOL_OUTSIDE) continue;
                if (fabs(local.z) > half + TOL_OUTSIDE) continue;
                
                // 至少一个轴必须超出 half（否则实际在内部，第一遍已处理）
                double dx_out = fabs(local.x) - half;
                double dy_out = fabs(local.y) - half;
                double dz_out = fabs(local.z) - half;
                double max_out = fmax(dx_out, fmax(dy_out, dz_out));
                if (max_out <= 0) continue;
                if (max_out > TOL_OUTSIDE) continue;  // 漂得太远，不处理
                
                if (max_out < near_dist) {
                    near_dist = max_out;
                    detected_incl = k;
                    is_just_outside = true;
                }
            }
        }

        if (!is_inside_or_on && !is_just_outside) continue;
        if (detected_incl < 0) continue;

        // ---- 找到最近面、计算投影位置和法向量 ----
        Vec3 c = inclusion_centers[detected_incl];
        Vec3 local = pos - c;

        // 选最近单面 + 投影（统一 helper）
        int face_sign[3];
        inclusion_nearest_face(local, face_sign);
        Vec3 best_proj   = inclusion_project_capture(local, face_sign, half, 30.0); // 30b margin

        // ---- 投影 + PIN ----
        Vec3 old_pos_upd = network->nodes[i].pos;
        network->nodes[i].pos        = network->cell.pbc_fold(c + best_proj);
        network->nodes[i].constraint = INCLUSION_NODE;
        network->nodes[i].v          = Vec3(0.0);
        { double jmp = (network->nodes[i].pos - old_pos_upd).norm();
          if (jmp > 500.0)
            ExaDiS_log("[FIX] src=update tag=(%d,%d) jump=%.0f from=(%.0f,%.0f,%.0f) to=(%.0f,%.0f,%.0f)\n",
                       network->nodes[i].tag.domain, network->nodes[i].tag.index, jmp,
                       old_pos_upd.x, old_pos_upd.y, old_pos_upd.z,
                       network->nodes[i].pos.x, network->nodes[i].pos.y, network->nodes[i].pos.z); }
        new_fixed++;
        new_projected++;
        if (is_just_outside) new_outside++;
    }

    // ============================================================
    // 第二步：清除幽灵段（保持原有逻辑作为兜底）
    // ============================================================
    int nsegs = network->number_of_segs();
    int ghost_segs = 0;
    for (int i = 0; i < nsegs; ++i) {
        int n1 = network->segs[i].n1;
        int n2 = network->segs[i].n2;
        if (network->nodes[n1].constraint != INCLUSION_NODE) continue;
        if (network->nodes[n2].constraint != INCLUSION_NODE) continue;

        bool n1_inside = is_node_strictly_inside_inclusion(network->nodes[n1].pos);
        bool n2_inside = is_node_strictly_inside_inclusion(network->nodes[n2].pos);
        if ((n1_inside || n2_inside) && network->segs[i].burg.norm2() > 1e-20) {
            network->segs[i].burg = Vec3(0.0);
            ghost_segs++;
        }
    }

    if (new_fixed > 0 || ghost_segs > 0) {
        total_fixed_count += new_fixed;
        ExaDiS_log("Step: new fixed=%d (projected=%d, outside=%d), total fixed=%d, ghost segs zeroed=%d\n",
                   new_fixed, new_projected, new_outside, total_fixed_count, ghost_segs);
        if (ghost_segs > 0) {
            ExaDiS_log("*** Orowan ring candidate: ghost segs = %d ***\n", ghost_segs);
        }
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::seg_cube_intersect()
 *
 *    计算线段 [p_out → p_in] 与轴对齐立方体的入射面交点。
 *
 *    修复：
 *      删除原来有缺陷的"棱角吸附"逻辑。
 *      原逻辑在棱线附近会把节点推到棱线上，导致节点偏离真实交点。
 *      修复后改为"最近面投影"：找到交点最靠近的那个面，
 *      只把该方向的坐标精确设为 ±half，其余坐标保持 slab 计算结果。
 *
 *-------------------------------------------------------------------------*/
bool System::seg_cube_intersect(const Vec3& p_out, const Vec3& p_in,
                                 const Vec3& center, double half, Vec3& hit) const
{
    Vec3 d = p_in - p_out;
    double tmin = 0.0, tmax = 1.0;
 
    for (int k = 0; k < 3; ++k) {
        double lo = center[k] - half;
        double hi = center[k] + half;
        if (fabs(d[k]) < 1e-14) {
            // 该方向平行：检查是否在 slab 内
            if (p_out[k] < lo || p_out[k] > hi) return false;
        } else {
            double t1 = (lo - p_out[k]) / d[k];
            double t2 = (hi - p_out[k]) / d[k];
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
 
    // tmin 对应从 p_out 出发第一个进入夹杂的面（入射面交点）
    hit = p_out + tmin * d;
 
    // 最近面投影：消除浮点误差，把 hit 精确放到最近的面上
    // 只修正最近面方向的坐标，其余坐标保持不变
    Vec3 local = hit - center;
    double min_face_dist = 1e30;
    int nearest_axis = 0;
    double nearest_sign = 1.0;
    for (int k = 0; k < 3; ++k) {
        double dist_pos = fabs(local[k] - half);
        double dist_neg = fabs(local[k] + half);
        if (dist_pos < min_face_dist) {
            min_face_dist = dist_pos;
            nearest_axis = k;
            nearest_sign = 1.0;
        }
        if (dist_neg < min_face_dist) {
            min_face_dist = dist_neg;
            nearest_axis = k;
            nearest_sign = -1.0;
        }
    }
    local[nearest_axis] = nearest_sign * half;
    hit = center + local;
 
    return true;
}
/*---------------------------------------------------------------------------
 *
 *    Function:     System::snap_nodes_to_surface()
 *
 *    在积分之前调用。
 *    对于即将进入夹杂的节点，沿其速度方向计算与夹杂表面的交点，
 *    将节点直接移动到交点处并固定为 PINNED_NODE。
 *
 *    优点：不产生新节点，不需要拓扑操作，节点数不增长。
 *    适用：节点在通道中运动，速度方向指向夹杂表面的情况。
 *
 *    调用时机：mobility->compute() 之后，integrator->integrate() 之前。
 *
 *-------------------------------------------------------------------------*/
void System::snap_nodes_to_surface(SerialDisNet* network)
{
    if (!inclusion_enabled) return;

    double half = inclusion_a_dim * 0.5;
int nnodes = network->number_of_nodes();
    int snapped = 0;

    for (int i = 0; i < nnodes; i++) {
        // 只处理非 PINNED 的自由节点
        if (network->nodes[i].constraint == INCLUSION_NODE) continue;

        Vec3 pos = network->nodes[i].pos;
        Vec3 vel = network->nodes[i].v;

        // 速度为零的节点不会移动，跳过
        double vn = vel.norm();
        if (vn < 1e-30) continue;

        // 如果节点已经在夹杂内，交给 insert_surface_nodes 处理
        if (is_node_in_inclusion(pos)) continue;

        // 沿速度方向延伸 maxseg 距离，预测节点的运动趋势
        // 不依赖 dt 的精确值，只判断速度方向是否指向夹杂
        Vec3 pos_next = pos + (params.maxseg / vn) * vel;

        // 检查延伸方向上是否会进入夹杂
        bool next_in = is_node_in_inclusion(pos_next);

        if (!next_in) continue;  // 速度方向不指向夹杂，跳过

        // 找到 pos → pos_next 与夹杂表面的交点
        double min_dist = 1e30;
        Vec3 best_hit;
        bool found = false;

        for (const auto& center : inclusion_centers) {
            Vec3 hit;
            if (seg_cube_intersect(pos, pos_next, center, half, hit)) {
                double dist = (hit - pos).norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    best_hit = hit;
                    found = true;
                }
            }
        }

        if (!found) continue;

        // 将节点吸附到表面交点并固定
        network->nodes[i].pos        = network->cell.pbc_fold(best_hit);
        network->nodes[i].constraint = INCLUSION_NODE;
        network->nodes[i].v          = Vec3(0.0);
        snapped++;
    }

    if (snapped > 0) {
        ExaDiS_log("Orowan: snapped %d nodes to inclusion surface\n", snapped);
        // 重建连接表（节点位置改变，但拓扑不变，只需刷新指针）
        network->update_ptr();
    }
}
/*---------------------------------------------------------------------------
 *
 *    Function:     System::get_face_normal()
 *                  返回交点 hit 所在立方体面的外法向量
 *
 *-------------------------------------------------------------------------*/
Vec3 System::get_face_normal(const Vec3& hit, const Vec3& center, double half) const
{
    Vec3 local = hit - center;
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

/*---------------------------------------------------------------------------
 *
 *    Function:     System::project_surface_node_velocity()
 *                  在每步积分前调用。
 *                  把所有表面节点的速度投影到它们所在的面上，
 *                  去掉法向分量，使节点只能在面内运动。
 *
 *-------------------------------------------------------------------------*/
void System::project_surface_node_velocity(SerialDisNet* network)
{
    if (!inclusion_enabled) return;

    int nnodes = network->number_of_nodes();
    for (int i = 0; i < nnodes; i++) {
        if (network->nodes[i].constraint != INCLUSION_NODE) continue;
        if (inclusion_on_edge(network->nodes[i].pos, 2.0)) {
            network->nodes[i].v = Vec3(0.0);
            continue;
        }
        Vec3 n_surface;
        if (inclusion_single_face_normal(network->nodes[i].pos, n_surface) < 0) continue;

        // Same connected non-ghost segment plane that correct_surface_node_positions
        // uses → velocity stays on the SAME 1D (face ∩ glide-plane) line as the position.
        Vec3 glide_n(0.0); bool has_glide = false;
        for (int j = 0; j < network->conn[i].num; j++) {
            int s = network->conn[i].seg[j];
            if (network->segs[s].burg.norm2() < 1e-20) continue;
            if (network->segs[s].plane.norm2() < 1e-10) continue;
            glide_n = network->segs[s].plane.normalized(); has_glide = true; break;
        }

        Vec3& v = network->nodes[i].v;
        if (has_glide) {
            Vec3 l = cross(glide_n, n_surface);
            double ln = l.norm();
            if (ln > 1e-6) { l = (1.0/ln) * l; v = dot(v, l) * l; }
            else           { v = v - dot(v, n_surface) * n_surface; }
        } else {
            v = v - dot(v, n_surface) * n_surface;
        }
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::check_surface_node_transition()
 *
 *    重新设计版本（PIN→SURFACE 改造，第二阶段）
 *
 *    根据节点的当前位置，重新判定它应该贴在哪些面上（可能是 1、2、或
 *    3 个面，分别对应面节点、棱节点、角节点），更新它的法向量和位置。
 *
 *    与旧版本的关键区别：
 *    - 不允许节点脱离表面恢复为自由节点（遵循设计文档 3A）
 *    - 正确处理棱节点和角节点（合成法向量）
 *    - 正确处理面↔棱、棱↔角之间的状态变化
 *
 *    本函数不处理"节点深入夹杂内部"的情况——那由
 *    update_inclusion_constraints 处理。
 *
 *-------------------------------------------------------------------------*/
void System::check_surface_node_transition(SerialDisNet* network)
{
    // Stage 3: disabled — face assignment now handled by correct_surface_node_positions
    // via realtime classify each step. Edge handling → stage 5.
    return;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::insert_surface_nodes()
 *
 *    【修复1】第二阶段改为处理任意数量的外侧邻居（不再只处理 size==2）
 *    【修复3】add_seg 前检查段长，防止 PBC 导致的超长段
 *
 *-------------------------------------------------------------------------*/
void System::insert_surface_nodes(SerialDisNet* network)
{
    if (!inclusion_enabled) return;
 
    double half = inclusion_a_dim * 0.5;
 
    // ================================================================
    // 第一阶段：找到所有跨越夹杂边界的线段，在交点处插入 PINNED 节点
    // ================================================================
 
    struct SplitInfo { int seg_id; Vec3 hit_pos; int incl_id; };
    std::vector<SplitInfo> to_split;

    int nsegs_initial = network->number_of_segs();
    for (int i = 0; i < nsegs_initial; i++) {
        int n1 = network->segs[i].n1;
        int n2 = network->segs[i].n2;
 
        bool n1_pinned = (network->nodes[n1].constraint == INCLUSION_NODE);
        bool n2_pinned = (network->nodes[n2].constraint == INCLUSION_NODE);
        if (n1_pinned && n2_pinned) continue;
 
        Vec3 p1 = network->nodes[n1].pos;
        Vec3 p2 = network->cell.pbc_position(p1, network->nodes[n2].pos);
 
        bool p1_in = is_node_in_inclusion(p1);
        bool p2_in = is_node_in_inclusion(p2);
        if (p1_in == p2_in) continue;
 
        int inner_node  = p1_in ? n1 : n2;
        Vec3 p_in      = p1_in ? p1 : p2;
        Vec3 p_out_raw  = p1_in ? p2 : p1;
 
        // 增量式过滤
        long long inner_key = network->nodes[inner_node].tag.domain * 1000000LL
                            + network->nodes[inner_node].tag.index;
        auto it = node_was_inside.find(inner_key);
        if (it != node_was_inside.end() && it->second) continue;
 
        bool n1_in = p1_in, n2_in = p2_in;
        if ((n1_pinned && n2_in) || (n2_pinned && n1_in)) continue;
 
        // 找 p_in 所属夹杂，以 p_in 为基准折叠 p_out
        Vec3 best_hit;
        bool found = false;
        int found_incl_id = -1;
        for (int incl_idx = 0; incl_idx < (int)inclusion_centers.size(); incl_idx++) {
            const Vec3& center = inclusion_centers[incl_idx];
            Vec3 local_in = p_in - center;
            bool p_in_here = (fabs(local_in.x) <= half * (1.0 + 1e-6) &&
                              fabs(local_in.y) <= half * (1.0 + 1e-6) &&
                              fabs(local_in.z) <= half * (1.0 + 1e-6));
            if (!p_in_here) continue;

            Vec3 p_out = network->cell.pbc_position(p_in, p_out_raw);
            Vec3 hit;
            if (seg_cube_intersect(p_out, p_in, center, half, hit)) {
                best_hit = hit;
                found = true;
                found_incl_id = incl_idx;
            }
            break;
        }

        if (found) to_split.push_back({i, best_hit, found_incl_id});
    }
 
    if (to_split.empty()) {
        node_was_inside.clear();
        for (int i = 0; i < network->number_of_nodes(); i++) {
            if (is_node_in_inclusion(network->nodes[i].pos)) {
                long long key = network->nodes[i].tag.domain * 1000000LL
                              + network->nodes[i].tag.index;
                node_was_inside[key] = true;
            }
        }
        return;
    }
 
    std::sort(to_split.begin(), to_split.end(),
              [](const auto& a, const auto& b){ return a.seg_id > b.seg_id; });
 
    int new_surface_nodes = 0;
    for (auto& info : to_split) {
        int seg_id   = info.seg_id;
        Vec3 hit_pos = info.hit_pos;
        int incl_id  = info.incl_id;
        if (seg_id >= network->number_of_segs()) continue;
 
        int n1 = network->segs[seg_id].n1;
        int n2 = network->segs[seg_id].n2;
        bool n1_pinned = (network->nodes[n1].constraint == INCLUSION_NODE);
        bool n2_pinned = (network->nodes[n2].constraint == INCLUSION_NODE);
        if (n1_pinned && n2_pinned) continue;
 
        bool n1_in = is_node_in_inclusion(network->nodes[n1].pos);
        bool n2_in = is_node_in_inclusion(network->nodes[n2].pos);
        if ((n1_pinned && n2_in) || (n2_pinned && n1_in)) continue;
 
        double tol2 = (0.05 * inclusion_a_dim) * (0.05 * inclusion_a_dim);
        bool already_pinned = false;
        if (n1_pinned && (network->nodes[n1].pos - hit_pos).norm2() < tol2)
            already_pinned = true;
        if (n2_pinned && (network->nodes[n2].pos - hit_pos).norm2() < tol2)
            already_pinned = true;
        if (already_pinned) continue;
 
        int new_node = network->split_seg(seg_id, hit_pos);
        if (new_node < 0) continue;

        network->nodes[new_node].constraint = INCLUSION_NODE;
        network->nodes[new_node].v = Vec3(0.0);
        new_surface_nodes++;
        // 验证插入位置的深度
        Vec3 inserted_pos = network->nodes[new_node].pos;
        Vec3 center = inclusion_centers[incl_id];
        Vec3 local = inserted_pos - center;
        double depth = fmin(fmin(
            half - fabs(local.x),
            half - fabs(local.y)),
            half - fabs(local.z));
        ExaDiS_log("Orowan: new surface node depth=%.4f b, hit_pos depth=%.4f b\n",
                depth,
                fmin(fmin(
                    half - fabs((hit_pos - center).x),
                    half - fabs((hit_pos - center).y)),
                    half - fabs((hit_pos - center).z)));
        long long new_key = network->nodes[new_node].tag.domain * 1000000LL
                          + network->nodes[new_node].tag.index;
        node_was_inside[new_key] = true;
    }
 
    if (new_surface_nodes > 0) {
        network->generate_connectivity();
        network->update_ptr();
        ExaDiS_log("Orowan: inserted %d surface nodes on inclusion boundary\n",
                   new_surface_nodes);
    }
 
    // ================================================================
    // 第二阶段：清除内部节点，重新连接外侧节点
    //
    // 【修复1】改为处理任意数量的外侧邻居：
    //   size==0：ci是孤立内部节点，删除即可
    //   size==1：ci是位错臂末端附近的内部节点，删除后留单臂悬端（正常）
    //   size==2：原有逻辑，检查穿越后连接
    //   size>=3：junction节点，对所有外侧邻居对两两检查并连接
    // ================================================================
 
    std::unordered_map<long long, int> tag_to_idx;
    auto rebuild_tag_map = [&]() {
        tag_to_idx.clear();
        tag_to_idx.reserve(network->number_of_nodes());
        for (int i = 0; i < network->number_of_nodes(); i++) {
            long long key = network->nodes[i].tag.domain * 1000000LL
                          + network->nodes[i].tag.index;
            tag_to_idx[key] = i;
        }
    };
 
    auto find_node_by_tag = [&](const NodeTag& tag) -> int {
        long long key = tag.domain * 1000000LL + tag.index;
        auto it = tag_to_idx.find(key);
        return (it != tag_to_idx.end()) ? it->second : -1;
    };
 
    // 段长合理性上限：超过此值说明 PBC 折叠方向有问题
    double max_seg_len = params.maxseg * 3.0;
 
    bool did_something = true;
    int iter_count = 0;
    int max_iter = network->number_of_nodes() + 10;
 
    network->generate_connectivity();
    rebuild_tag_map();
 
    while (did_something && iter_count < max_iter) {
        did_something = false;
        iter_count++;
 
        // 找第一个严格在夹杂内部的非 PINNED 节点
        int ci = -1;
        for (int i = 0; i < network->number_of_nodes(); i++) {
            if (network->nodes[i].constraint == INCLUSION_NODE) continue;
            if (is_node_strictly_inside_inclusion(network->nodes[i].pos)) {
                ci = i;
                break;
            }
        }
        if (ci < 0) break;
        did_something = true;
 
        struct NeighborInfo {
            NodeTag tag;
            Vec3 burg;
            Vec3 plane;
        };
        std::vector<NeighborInfo> outer_neighbors;
 
        for (int j = 0; j < network->conn[ci].num; j++) {
            int nb     = network->conn[ci].node[j];
            int seg_id = network->conn[ci].seg[j];
            int order  = network->conn[ci].order[j];
 
            bool nb_is_inner =
                (network->nodes[nb].constraint != INCLUSION_NODE) &&
                is_node_strictly_inside_inclusion(network->nodes[nb].pos);
 
            if (!nb_is_inner) {
                NeighborInfo info;
                info.tag   = network->nodes[nb].tag;
                info.burg  = order * network->segs[seg_id].burg;
                info.plane = network->segs[seg_id].plane;
                outer_neighbors.push_back(info);
            }
        }
 
        // 删除内部节点
        network->remove_nodes({ci});
        network->generate_connectivity();
        network->update_ptr();
 
        // 孤立节点保护：删除 ci 后连接数 <=1 的非 PINNED 节点固定
        for (int i = 0; i < network->number_of_nodes(); i++) {
            if (network->nodes[i].constraint == INCLUSION_NODE) continue;
            if (network->conn[i].num <= 1 && is_node_in_inclusion(network->nodes[i].pos)) {
                ExaDiS_log("[FIX] src=isolate980 tag=(%d,%d) conn=%d pos=(%.0f,%.0f,%.0f)\n",
                           network->nodes[i].tag.domain, network->nodes[i].tag.index,
                           network->conn[i].num,
                           network->nodes[i].pos.x, network->nodes[i].pos.y, network->nodes[i].pos.z);
                network->nodes[i].constraint = INCLUSION_NODE;
                network->nodes[i].v = Vec3(0.0);
                long long key = network->nodes[i].tag.domain * 1000000LL
                              + network->nodes[i].tag.index;
                node_was_inside[key] = true;
            }
        }
 
        rebuild_tag_map();
 
        // 【修复1】对所有外侧邻居对两两检查并连接
        // size==0 或 size==1：不需要连接
        // size==2：一对，原有逻辑
        // size>=3：junction，多对两两处理
        int n_outer = (int)outer_neighbors.size();
        for (int a = 0; a < n_outer; a++) {
            for (int b = a + 1; b < n_outer; b++) {
 
                int na     = find_node_by_tag(outer_neighbors[a].tag);
                int nb_idx = find_node_by_tag(outer_neighbors[b].tag);
 
                if (na < 0 || nb_idx < 0) {
                    ExaDiS_log("Warning: Orowan outer neighbor not found by tag, skipping\n");
                    continue;
                }
                if (network->find_connection(na, nb_idx) >= 0) continue;
 
                Vec3 pa = network->nodes[na].pos;
                Vec3 pb = network->cell.pbc_position(pa, network->nodes[nb_idx].pos);
 
                // 【修复3】检查段长，防止 PBC 导致的超长段
                double seg_len = (pb - pa).norm();
                if (seg_len > max_seg_len) {
                    ExaDiS_log("Warning: Orowan skipping too-long segment (%.0f b), "
                               "likely PBC issue\n", seg_len);
                    continue;
                }
 
                // 穿越检查：只有至少一端在夹杂内才检查
                bool na_in = is_node_in_inclusion(pa);
                bool nb_in = is_node_in_inclusion(pb);
                bool crosses = false;
                if (na_in || nb_in) {
                    for (const auto& center : inclusion_centers) {
                        Vec3 hit;
                        if (seg_cube_intersect(pa, pb, center, half, hit)) {
                            crosses = true;
                            break;
                        }
                    }
                }
 
                if (crosses) {
                    ExaDiS_log("Orowan: two arms on opposite sides, not connecting\n");
                    continue;
                }
 
                // 正常连接：使用 outer_neighbors[b] 的 Burgers 矢量
                network->add_seg(na, nb_idx, outer_neighbors[b].burg,
                                 outer_neighbors[b].plane);
                network->generate_connectivity();
                network->update_ptr();
                rebuild_tag_map();
            }
        }
    }
 
    if (iter_count >= max_iter) {
        ExaDiS_log("Warning: insert_surface_nodes hit max iteration limit (%d)\n",
                   max_iter);
    }
 
    network->generate_connectivity();
    network->update_ptr();
 
    // 更新 node_was_inside 状态记录
    node_was_inside.clear();
    for (int i = 0; i < network->number_of_nodes(); i++) {
        if (is_node_in_inclusion(network->nodes[i].pos)) {
            long long key = network->nodes[i].tag.domain * 1000000LL
                          + network->nodes[i].tag.index;
            node_was_inside[key] = true;
        }
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::reset_glide_planes()
 *
 *-------------------------------------------------------------------------*/
void System::reset_glide_planes()
{
    if (!crystal.use_glide_planes) return;

    DeviceDisNet* net = get_device_network();

    int err = 0;
    Kokkos::parallel_reduce(net->Nsegs_local, KOKKOS_LAMBDA(const int& i, int& err) {
        auto segs = net->get_segs();
        if (segs[i].plane.norm2() < 1e-10) err += 1;
    }, err);
    Kokkos::fence();
    if (err > 0)
        ExaDiS_fatal("Error: %d segments have no glide plane assigned\n", err);

    if (!crystal.enforce_glide_planes) return;

    bool use_oprec = (oprec);
    if (use_oprec) use_oprec = oprec->record;
    Kokkos::DualView<int*> flag;
    Kokkos::DualView<Vec3*> val;
    Kokkos::DualView<NodeTag*[2]> tags;
    if (use_oprec) {
        int N_local = MAX(net->Nnodes_local, net->Nsegs_local);
        flag = Kokkos::DualView<int*>("flag", N_local);
        val = Kokkos::DualView<Vec3*>("val", N_local);
        tags = Kokkos::DualView<NodeTag*[2]>("tags", N_local);
    }

    if (xold.extent(0) > 0) {
        T_x& xprev = xold;
        if (use_oprec) Kokkos::deep_copy(flag.d_view, 0);
        Kokkos::parallel_for(net->Nnodes_local, KOKKOS_LAMBDA(const int& i) {
            auto nodes = net->get_nodes();
            if (nodes[i].constraint == PINNED_NODE || nodes[i].constraint == SURFACE_NODE 
                || nodes[i].constraint == INCLUSION_NODE) return;
            auto segs = net->get_segs();
            auto conn = net->get_conn();
            auto cell = net->cell;

            int numplanes = 0;
            Vec3 planes[MAX_CONN];
            for (int j = 0; j < conn[i].num; j++) {
                int s = conn[i].seg[j];
                Vec3 p = segs[s].plane;
                if (j == 0) {
                    planes[0] = p;
                    numplanes = 1;
                } else {
                    for (int k = 0; k < numplanes; k++)
                        p = p.orthogonalize(planes[k]);
                    if (p.norm2() > 1e-5)
                        planes[numplanes++] = p.normalized();
                }
            }

            Vec3 pold = xprev(i);
            Vec3 pcur = nodes[i].pos;
            Vec3 p = cell.pbc_position(xprev(i), pcur);
            if (numplanes == 1) {
                double eqn = dot(planes[0], p) - dot(planes[0], pold);
                p -= eqn * planes[0];
            } else if (numplanes == 2) {
                Vec3 l = cross(planes[0], planes[1]).normalized();
                Vec3 dr = p - pold;
                p = pold + dot(l, dr) * l;
            } else {
                p = pold;
            }
            if ((p-pcur).norm2() > 1e-5) {
                p = cell.pbc_fold(p);
                nodes[i].pos = p;
                if (use_oprec) {
                    flag.d_view(i) = 1;
                    val.d_view(i) = p;
                    tags.d_view(i,0) = nodes[i].tag;
                }
            }
        });
        Kokkos::fence();

        if (use_oprec) {
            Kokkos::deep_copy(flag.h_view, flag.d_view);
            Kokkos::deep_copy(val.h_view, val.d_view);
            Kokkos::deep_copy(tags.h_view, tags.d_view);
            for (int i = 0; i < net->Nnodes_local; i++) {
                if (!flag.h_view(i)) continue;
                oprec->add_op(OpRec::MoveNode(tags.h_view(i,0), val.h_view(i)));
            }
        }
    }

    Crystal* cryst = &crystal;
    if (use_oprec) Kokkos::deep_copy(flag.d_view, 0);
    Kokkos::parallel_for(net->Nsegs_local, KOKKOS_LAMBDA(const int& i) {
        auto nodes = net->get_nodes();
        auto segs = net->get_segs();
        if (nodes[segs[i].n1].constraint == INCLUSION_NODE ||
            nodes[segs[i].n2].constraint == INCLUSION_NODE) return;
        Vec3 pold = segs[i].plane;
        Vec3 pnew = cryst->find_seg_glide_plane(net, i);
        if (pnew.norm2() > 1e-5 && (pnew-pold).norm2() > 1e-5) {
            segs[i].plane = pnew;
            if (use_oprec) {
                flag.d_view(i) = 1;
                val.d_view(i) = pnew;
                tags.d_view(i,0) = nodes[segs[i].n1].tag;
                tags.d_view(i,1) = nodes[segs[i].n2].tag;
            }
        }
    });
    Kokkos::fence();

    if (use_oprec) {
        Kokkos::deep_copy(flag.h_view, flag.d_view);
        Kokkos::deep_copy(val.h_view, val.d_view);
        Kokkos::deep_copy(tags.h_view, tags.d_view);
        for (int i = 0; i < net->Nsegs_local; i++) {
            if (!flag.h_view(i)) continue;
            oprec->add_op(OpRec::UpdateSegPlane(tags.h_view(i,0), tags.h_view(i,1), val.h_view(i)));
        }
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::write_config()
 *
 *-------------------------------------------------------------------------*/
void System::write_config(std::string filename)
{
#ifdef MPI
    ExaDiS_fatal("System::write_config not implemented for MPI\n");
#endif

    Kokkos::fence();
    timer[TIMER_OUTPUT].start();

    int active_net = net_mngr->get_active();
    SerialDisNet *network = get_serial_network();
    network->write_data(filename);

    net_mngr->set_active(active_net);

    Kokkos::fence();
    timer[TIMER_OUTPUT].stop();
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::print_timers()
 *
 *-------------------------------------------------------------------------*/
void System::print_timers(double timetot, bool dev)
{
    if (timetot < 0.0) {
        timetot = 0.0;
        for (int i = 0; i < TIMER_END; i++)
            timetot += timer[i].accumtime;
    }
    double ftime[TIMER_END];
    for (int i = 0; i < TIMER_END; i++)
        ftime[i] = (timetot > 0.0) ? timer[i].accumtime/timetot*100.0 : 0.0;

    ExaDiS_log("----------------------------------------------\n");
    ExaDiS_log("%-20s %11.3f sec (%.2f%%)\n", "Force time:", timer[TIMER_FORCE].accumtime, ftime[TIMER_FORCE]);
    ExaDiS_log("%-20s %11.3f sec (%.2f%%)\n", "Mobility time:", timer[TIMER_MOBILITY].accumtime, ftime[TIMER_MOBILITY]);
    ExaDiS_log("%-20s %11.3f sec (%.2f%%)\n", "Integration time:", timer[TIMER_INTEGRATION].accumtime, ftime[TIMER_INTEGRATION]);
    ExaDiS_log("%-20s %11.3f sec (%.2f%%)\n", "Cross-slip time:", timer[TIMER_CROSSSLIP].accumtime, ftime[TIMER_CROSSSLIP]);
    ExaDiS_log("%-20s %11.3f sec (%.2f%%)\n", "Collision time:", timer[TIMER_COLLISION].accumtime, ftime[TIMER_COLLISION]);
    ExaDiS_log("%-20s %11.3f sec (%.2f%%)\n", "Topology time:", timer[TIMER_TOPOLOGY].accumtime, ftime[TIMER_TOPOLOGY]);
    ExaDiS_log("%-20s %11.3f sec (%.2f%%)\n", "Remesh time:", timer[TIMER_REMESH].accumtime, ftime[TIMER_REMESH]);
    ExaDiS_log("%-20s %11.3f sec (%.2f%%)\n", "Output time:", timer[TIMER_OUTPUT].accumtime, ftime[TIMER_OUTPUT]);
    ExaDiS_log("----------------------------------------------\n");
    if (dev && numdevtimer > 0) {
        for (int i = 0; i < numdevtimer; i++)
            ExaDiS_log("%s time: %.3f sec\n", devtimer[i].label.c_str(), devtimer[i].accumtime);
        ExaDiS_log("----------------------------------------------\n");
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     make_system()
 *
 *-------------------------------------------------------------------------*/
System* make_system(SerialDisNet* net, Crystal crystal, Params params) {
    System* system = exadis_new<System>();
    system->initialize(params, crystal, net);
    return system;
}

/*---------------------------------------------------------------------------
 *
 *    Function:     make_network_manager()
 *
 *-------------------------------------------------------------------------*/
DisNetManager* make_network_manager(SerialDisNet* net) {
    net->generate_connectivity();
    net->update_ptr();
    return exadis_new<DisNetManager>(net);
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::same_inclusion_face()
 *
 *-------------------------------------------------------------------------*/
bool System::same_inclusion_face(const Vec3& pos1, const Vec3& pos2) const {
    if (!inclusion_enabled) return false;
    double half = inclusion_a_dim * 0.5;
    double tol = 0.01 * inclusion_a_dim;  // 从 5% 收紧到 1%

    for (const auto& center : inclusion_centers) {
        for (int axis = 0; axis < 3; axis++) {
            double c1 = pos1[axis] - center[axis];
            double c2 = pos2[axis] - center[axis];

            bool p1_on_face = fabs(fabs(c1) - half) < tol;
            bool p2_on_face = fabs(fabs(c2) - half) < tol;
            bool same_side  = (c1 * c2 > 0.0);  // 同侧（同号）

            if (p1_on_face && p2_on_face && same_side) {
                // 确认两点在其余两个方向上均在夹杂范围内
                bool in_face_1 = true, in_face_2 = true;
                for (int other = 0; other < 3; other++) {
                    if (other == axis) continue;
                    if (fabs(pos1[other] - center[other]) > half + tol) in_face_1 = false;
                    if (fabs(pos2[other] - center[other]) > half + tol) in_face_2 = false;
                }
                if (in_face_1 && in_face_2) return true;
            }
        }
    }
    return false;
}

void System::detect_orowan_loop(SerialDisNet* network)
{
    if (!inclusion_enabled) return;

    for (int i = 0; i < network->number_of_segs(); i++) {
        if (network->segs[i].burg.norm2() < 1e-20) {
            int n1 = network->segs[i].n1;
            int n2 = network->segs[i].n2;
            if (network->nodes[n1].constraint == INCLUSION_NODE &&
                network->nodes[n2].constraint == INCLUSION_NODE) {
                orowan_loop_count++;
                ExaDiS_log("Orowan loop detected! Total count: %d\n", orowan_loop_count);
                return;  // 每步最多记录一次
            }
        }
    }
}
/*---------------------------------------------------------------------------
 *
 *    Function:     System::enforce_edge_continuity()
 *
 *    棱穿透修复：
 *    扫描所有线段，找到两端都是表面节点但位于不同面的段。
 *    如果段穿越夹杂内部（中点在内），在段所跨的棱上插入一个新的
 *    PINNED 表面节点，将段一分为二。
 *
 *    判据：
 *    - 两端都在 surface_node_normal 表中
 *    - 两端属于同一个夹杂
 *    - 两端的法向量不同（在不同面上）
 *    - 段的中点位于该夹杂内部
 *
 *    新节点：
 *    - 位置：投影到 n1 和 n2 共享的棱上，沿棱方向取段中点的坐标
 *    - 法向量：(n1 + n2) 归一化（暂用近似，待 SURFACE 改造时细化）
 *    - constraint：INCLUSION_NODE（与现有表面节点一致）
 *    - 登记到 surface_node_normal 和 surface_node_incl_id
 *
 *    调用时机：在 step() 中所有 update_inclusion_constraints 之后调用。
 *
 *-------------------------------------------------------------------------*/
void System::enforce_edge_continuity(SerialDisNet* network)
{
    return; // retired: no longer inserts map-tracked edge nodes
}
/*---------------------------------------------------------------------------
 *
 *    Function:     System::correct_surface_node_positions()
 *
 *    每步 integrate 之后调用。扫描所有表面节点，把它们的位置校正回
 *    严格的表面位置，去除数值漂移。
 *
 *    校正规则：
 *    - 法向量 n 的某个分量 |n[k]| > 0.5：强制 local[k] = sign(n[k]) * half
 *    - 法向量 n 的某个分量 |n[k]| ≤ 0.5：该轴自由（保留 local[k] 原值）
 *
 *-------------------------------------------------------------------------*/
void System::correct_surface_node_positions(SerialDisNet* network)
{
    if (!inclusion_enabled) return;
    if (network == nullptr) return;
    if (inclusion_centers.empty()) return;

    double half = inclusion_a_dim * 0.5;

    int nnodes = network->number_of_nodes();
    for (int i = 0; i < nnodes; i++) {
        if (network->nodes[i].constraint != INCLUSION_NODE) continue;
        int incl_id = -1; double bestd2 = 1e30;
        for (int k = 0; k < (int)inclusion_centers.size(); k++) {
            Vec3 d = network->nodes[i].pos - inclusion_centers[k];
            double dd = dot(d, d);
            if (dd < bestd2) { bestd2 = dd; incl_id = k; }
        }
        if (incl_id < 0) continue;
        Vec3 center = inclusion_centers[incl_id];

        // Realtime classify: face assignment from CURRENT position, never stale stored normal.
        Vec3 old_pos = network->nodes[i].pos;
        Vec3 local = network->cell.pbc_position(center, old_pos) - center;
        int face_sign[3];
        inclusion_nearest_face(local, face_sign);

        // Stage 6: get this node's glide plane from a connected non-ghost segment,
        // then project position onto the (face ∩ glide-plane) 1D line so the node
        // crawls along the slip line and naturally anchors when it hits a box edge.
        // Fall back to the 2D capture projection if no glide plane is available.
        Vec3 glide_n(0.0);
        bool has_glide = false;
        for (int j = 0; j < network->conn[i].num; j++) {
            int s = network->conn[i].seg[j];
            if (network->segs[s].burg.norm2() < 1e-20) continue;   // skip ghost segs
            if (network->segs[s].plane.norm2() < 1e-10) continue;  // skip planeless segs
            glide_n = network->segs[s].plane.normalized();
            has_glide = true;
            break;
        }
        // Sub-type by position: two axes near ±half → edge/corner node, pin to π∩E.
        int near_half = (fabs(local.x) >= half-2.0) + (fabs(local.y) >= half-2.0)
                      + (fabs(local.z) >= half-2.0);
        Vec3 proj;
        if (near_half >= 2)
            proj = inclusion_project_edge_point(local, half, glide_n);
        else
            proj = has_glide
                 ? inclusion_project_glide_line(local, face_sign, half, glide_n)
                 : inclusion_project_capture(local, face_sign, half, 30.0);
        Vec3 new_pos = network->cell.pbc_fold(center + proj);

        double drift = (new_pos - old_pos).norm();
        if (drift > 500.0)
            ExaDiS_log("[FIX] src=correct_NEW tag=(%d,%d) jump=%.0f from=(%.0f,%.0f,%.0f) to=(%.0f,%.0f,%.0f)\n",
                       network->nodes[i].tag.domain, network->nodes[i].tag.index, drift,
                       old_pos.x, old_pos.y, old_pos.z, new_pos.x, new_pos.y, new_pos.z);
        network->nodes[i].pos = new_pos;

    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     System::insert_edge_nodes()
 *
 *    Segment-driven edge interaction: for each segment with a c9 face node
 *    at one end, detect if the segment crosses a cube edge and insert a new
 *    c9 corner node at the π∩E intersection (glide plane ∩ cube edge).
 *
 *-------------------------------------------------------------------------*/
void System::insert_edge_nodes(SerialDisNet* network)
{
    if (!inclusion_enabled || inclusion_centers.empty()) return;
    double half = inclusion_a_dim * 0.5;
    const double min_sep = 5.0;

    int nsegs = network->number_of_segs();
    bool updated = false;

    for (int i = 0; i < nsegs; i++) {
        if (network->segs[i].burg.norm2() < 1e-20) continue;
        if (network->segs[i].plane.norm2() < 1e-10) continue;

        int n1 = network->segs[i].n1, n2 = network->segs[i].n2;
        bool c1 = (network->nodes[n1].constraint == INCLUSION_NODE);
        bool c2 = (network->nodes[n2].constraint == INCLUSION_NODE);
        if (!c1 && !c2) continue;

        // Use the c9 face node as anchor (na); skip if na is already on an edge
        int na = c1 ? n1 : n2, nb = c1 ? n2 : n1;
        if (inclusion_on_edge(network->nodes[na].pos, 2.0)) continue;

        // Nearest inclusion center to na
        int incl = 0; double bestd = 1e30;
        for (int k = 0; k < (int)inclusion_centers.size(); k++) {
            Vec3 dd = network->nodes[na].pos - inclusion_centers[k];
            double q = dot(dd, dd);
            if (q < bestd) { bestd = q; incl = k; }
        }
        Vec3 C = inclusion_centers[incl];
        Vec3 l1 = network->cell.pbc_position(C, network->nodes[na].pos) - C;
        Vec3 l2 = network->cell.pbc_position(C, network->nodes[nb].pos) - C;
        Vec3 n  = network->segs[i].plane.normalized();

        Vec3 xcut;
        if (!inclusion_segment_edge_cross(l1, l2, n, half, xcut)) continue;

        Vec3 pos = network->cell.pbc_fold(C + xcut);

        if ((pos - network->nodes[na].pos).norm() < min_sep) continue;
        if ((pos - network->nodes[nb].pos).norm() < min_sep) continue;

        int nnew = network->split_seg(i, pos);
        if (nnew < 0) continue;
        network->nodes[nnew].constraint = INCLUSION_NODE;
        network->nodes[nnew].v = Vec3(0.0);
        updated = true;
    }

    if (updated) {
        network->generate_connectivity();
        network->update_ptr();
    }
}

} // namespace ExaDiS
