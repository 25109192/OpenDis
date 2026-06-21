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
 *    接触式捕获：只捕获已进入、贴在表面、或表面外极近(接触余量
 *    CONTACT_MARGIN=5b 内)的节点。不再做"表面外 50b 吸引式"捕获
 *    ——位错该不该靠近、停在哪由力平衡决定，避免抢在弹性排斥前
 *    把节点拽到面上造成同源环重合。
 *
 *    被捕获的节点会被投影到最近的夹杂面、PIN、登记。
 *
 *-------------------------------------------------------------------------*/
void System::update_inclusion_constraints(SerialDisNet* network) {
    if (!inclusion_enabled) return;
    if (network == nullptr) return;

    int nnodes = network->number_of_nodes();
    static int total_fixed_count = 0;
    int new_fixed = 0;
    int new_projected = 0;
    double half = inclusion_a_dim * 0.5;

    const double CONTACT_MARGIN = 5.0;  // 接触判定余量(b):抓贴面但小幅偏出的节点

    // ============================================================
    // 第一步：扫描节点，捕获在内部、表面上、或表面外极近的节点
    // ============================================================
    for (int i = 0; i < nnodes; ++i) {
        if (network->nodes[i].constraint == PINNED_NODE) continue;
        // 救而不删:已在表面的 c9 照旧跳过;漂进夹杂内部的 c9 不跳过,落到下面用
        // glide_line 归位回表面(保 d),从而循环2 扫到时已无"内部端点",其段不被清零删掉。
        if (network->nodes[i].constraint == INCLUSION_NODE &&
            !is_node_strictly_inside_inclusion(network->nodes[i].pos)) continue;

        Vec3 pos = network->nodes[i].pos;

        // ---- 判定节点应该被处理的具体夹杂和情形 ----
        int  detected_incl  = -1;
        bool is_inside_or_on = false;

        // 第一遍：查找"在内或在表面"的夹杂
        for (int k = 0; k < (int)inclusion_centers.size(); k++) {
            Vec3 local = pos - inclusion_centers[k];
            if (fabs(local.x) <= half + CONTACT_MARGIN &&
                fabs(local.y) <= half + CONTACT_MARGIN &&
                fabs(local.z) <= half + CONTACT_MARGIN) {
                is_inside_or_on = true;
                detected_incl = k;
                break;
            }
        }

        if (!is_inside_or_on) continue;
        if (detected_incl < 0) continue;

        // ---- 找到最近面、计算投影位置和法向量 ----
        Vec3 c = inclusion_centers[detected_incl];
        Vec3 local = pos - c;

        // 选最近单面
        int face_sign[3];
        inclusion_nearest_face(local, face_sign);

        // 捕获前节点仍在真实滑移面内(ExaDiS 滑移约束保证),沿面法向硬摁会把它
        // 踢离真平面(Δd = glide_n[面轴]×穿透深度)。改用保 d 的 glide_line 投影,
        // 落点严格在 (面 ∩ 滑移面) 交线上,与 correct_surface_node_positions 一致。
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
        Vec3 best_proj = has_glide
                       ? inclusion_project_glide_line(local, face_sign, half, glide_n)
                       : inclusion_project_capture(local, face_sign, half, 30.0); // 无滑移面时退回

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
            ExaDiS_log("[DELMARK uic-zero] (%.0f,%.0f,%.0f) c=%d -(%.0f,%.0f,%.0f) c=%d in1=%d in2=%d\n",
                       network->nodes[n1].pos.x, network->nodes[n1].pos.y, network->nodes[n1].pos.z,
                       network->nodes[n1].constraint,
                       network->nodes[n2].pos.x, network->nodes[n2].pos.y, network->nodes[n2].pos.z,
                       network->nodes[n2].constraint, (int)n1_inside, (int)n2_inside);
        }
    }

    if (new_fixed > 0 || ghost_segs > 0) {
        total_fixed_count += new_fixed;
        ExaDiS_log("Step: new fixed=%d (projected=%d), total fixed=%d, ghost segs zeroed=%d\n",
                   new_fixed, new_projected, total_fixed_count, ghost_segs);
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

static inline void snap_to_nearest_face(Vec3& hit, const Vec3& center, double half) {
    Vec3 local = hit - center;
    double mind = 1e30; int axis = 0; double sign = 1.0;
    for (int k = 0; k < 3; ++k) {
        double dp = fabs(local[k] - half), dn = fabs(local[k] + half);
        if (dp < mind) { mind = dp; axis = k; sign =  1.0; }
        if (dn < mind) { mind = dn; axis = k; sign = -1.0; }
    }
    local[axis] = sign * half;
    hit = center + local;
}

/*---------------------------------------------------------------------------
 *    Function:     System::seg_cube_clip()
 *    两端在外、弦横切夹杂(角/棱斜入射)时,返回切入点 hit_in 和切出点 hit_out。
 *    要求 0 < tmin < tmax < 1(两端严格在外)。
 *-------------------------------------------------------------------------*/
bool System::seg_cube_clip(const Vec3& pa, const Vec3& pb,
                           const Vec3& center, double half,
                           Vec3& hit_in, Vec3& hit_out) const
{
    Vec3 d = pb - pa;
    double tmin = 0.0, tmax = 1.0;
    for (int k = 0; k < 3; ++k) {
        double lo = center[k] - half;
        double hi = center[k] + half;
        if (fabs(d[k]) < 1e-14) {
            if (pa[k] < lo || pa[k] > hi) return false;
        } else {
            double t1 = (lo - pa[k]) / d[k];
            double t2 = (hi - pa[k]) / d[k];
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
    const double eps = 1e-6;
    if (tmin <= eps || tmax >= 1.0 - eps || tmin >= tmax) return false;
    hit_in  = pa + tmin * d;
    hit_out = pa + tmax * d;
    snap_to_nearest_face(hit_in,  center, half);
    snap_to_nearest_face(hit_out, center, half);
    return true;
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
    struct ClipInfo { int seg_id; Vec3 hit_in; Vec3 hit_out; int incl_id; };
    std::vector<ClipInfo> to_clip;

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

        if (p1_in && p2_in) continue;  // 全在内部,交给第二阶段删除

        if (!p1_in && !p2_in) {
            // 切入:两端在外、弦横切夹杂(角/棱斜入射)
            for (int incl_idx = 0; incl_idx < (int)inclusion_centers.size(); incl_idx++) {
                Vec3 hin, hout;
                if (seg_cube_clip(p1, p2, inclusion_centers[incl_idx], half, hin, hout)) {
                    to_clip.push_back({i, hin, hout, incl_idx});
                    break;
                }
            }
            continue;
        }
        // 以下:一端在内一端在外(原有单点逻辑)

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
        long long new_key = network->nodes[new_node].tag.domain * 1000000LL
                          + network->nodes[new_node].tag.index;
        node_was_inside[new_key] = true;
    }

    // 切入双点捕获:两端在外、弦横切夹杂,在切入点和切出点各插一个 c9 节点
    std::sort(to_clip.begin(), to_clip.end(),
              [](const auto& a, const auto& b){ return a.seg_id > b.seg_id; });
    int clip_corner = 0;  // [EDGEPROBE] clip 段上插入的棱接触节点数
    for (auto& info : to_clip) {
        int seg_id = info.seg_id;
        if (seg_id >= network->number_of_segs()) continue;
        int cn1 = network->segs[seg_id].n1;
        int cn2 = network->segs[seg_id].n2;
        if (network->nodes[cn1].constraint == INCLUSION_NODE &&
            network->nodes[cn2].constraint == INCLUSION_NODE) continue;

        int nodeIn = network->split_seg(seg_id, info.hit_in);
        if (nodeIn < 0) continue;
        int s2 = network->number_of_segs() - 1;  // split_seg 追加远端半段在末尾
        int nodeOut = network->split_seg(s2, info.hit_out);

        network->nodes[nodeIn].constraint = INCLUSION_NODE;
        network->nodes[nodeIn].v = Vec3(0.0);
        new_surface_nodes++;
        long long kin = network->nodes[nodeIn].tag.domain * 1000000LL
                      + network->nodes[nodeIn].tag.index;
        node_was_inside[kin] = true;

        if (nodeOut >= 0) {
            network->nodes[nodeOut].constraint = INCLUSION_NODE;
            network->nodes[nodeOut].v = Vec3(0.0);
            new_surface_nodes++;
            long long kout = network->nodes[nodeOut].tag.domain * 1000000LL
                           + network->nodes[nodeOut].tag.index;
            node_was_inside[kout] = true;

            // ── 当场消除穿心截:在切入/切出点之间插棱接触节点 ──
            // 切入面与切出面共享一条立方体棱 E;接触点 = 段滑移面 π ∩ E。
            // s2 此刻是 nodeIn→nodeOut 这条穿心截,把它在 corner 处折开,
            // 使 nodeIn→corner、corner→nodeOut 各自贴在一个面上、不进内部。
            Vec3 C = inclusion_centers[info.incl_id];
            Vec3 lin  = info.hit_in  - C;
            Vec3 lout = info.hit_out - C;
            int ain = 0;
            if (fabs(lin[1])  > fabs(lin[ain]))  ain  = 1;
            if (fabs(lin[2])  > fabs(lin[ain]))  ain  = 2;
            int aout = 0;
            if (fabs(lout[1]) > fabs(lout[aout])) aout = 1;
            if (fabs(lout[2]) > fabs(lout[aout])) aout = 2;
            Vec3 npl = network->segs[s2].plane;
            // 仅相邻面(不同轴)才有共享棱可绕;同面/对面(同轴)跳过
            if (ain != aout && npl.norm2() > 1e-10) {
                Vec3 nrm = npl.normalized();
                int afree = 3 - ain - aout;
                double d  = dot(nrm, lin);              // lin 在滑移面 π 上
                double si = (lin[ain]   >= 0.0) ? half : -half;
                double so = (lout[aout] >= 0.0) ? half : -half;
                Vec3 corner(0.0);
                corner[ain]  = si;
                corner[aout] = so;
                if (fabs(nrm[afree]) > 1e-9)
                    corner[afree] = (d - nrm[ain]*si - nrm[aout]*so) / nrm[afree];
                else
                    corner[afree] = 0.5*(lin[afree] + lout[afree]);
                if (corner[afree] >  half) corner[afree] =  half;
                if (corner[afree] < -half) corner[afree] = -half;
                // 退化保护:接触点与某端点几乎重合则不插(避免零长段)
                if ((corner - lin).norm() > 1.0 && (corner - lout).norm() > 1.0) {
                    int nodeE = network->split_seg(s2, C + corner);
                    if (nodeE >= 0) {
                        network->nodes[nodeE].constraint = INCLUSION_NODE;
                        network->nodes[nodeE].v = Vec3(0.0);
                        new_surface_nodes++;
                        clip_corner++;
                        long long ke = network->nodes[nodeE].tag.domain * 1000000LL
                                     + network->nodes[nodeE].tag.index;
                        node_was_inside[ke] = true;
                    }
                }
            }
        }
    }
    if (!to_clip.empty())
        ExaDiS_log("[EDGEPROBE] clip segs=%zu, edge-corner inserted=%d\n",
                   to_clip.size(), clip_corner);

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
        for (int j = 0; j < network->conn[ci].num; j++) {
            int nb = network->conn[ci].node[j];
            ExaDiS_log("[DELMARK insSN-rmnode] ci=(%.0f,%.0f,%.0f) c=%d nb=(%.0f,%.0f,%.0f) c=%d\n",
                       network->nodes[ci].pos.x, network->nodes[ci].pos.y, network->nodes[ci].pos.z,
                       network->nodes[ci].constraint,
                       network->nodes[nb].pos.x, network->nodes[nb].pos.y, network->nodes[nb].pos.z,
                       network->nodes[nb].constraint);
        }
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
                    ExaDiS_log("[DELMARK insSN-toolong] (%.0f,%.0f,%.0f)-(%.0f,%.0f,%.0f) len=%.0f\n",
                               pa.x, pa.y, pa.z, pb.x, pb.y, pb.z, seg_len);
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
                    ExaDiS_log("[DELMARK insSN-nocon] (%.0f,%.0f,%.0f)-(%.0f,%.0f,%.0f)\n",
                               pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
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
    // [EDGEPROBE] 分支命中计数(每次调用一行)
    int p_anchor_edge=0, p_nocross=0, p_vertex=0, p_minsep=0, p_insert=0;
    int p_c9chord=0, p_c9corner=0;   // C9-C9 穿心弦:检出 / 成功插角

    for (int i = 0; i < nsegs; i++) {
        if (network->segs[i].burg.norm2() < 1e-20) continue;
        if (network->segs[i].plane.norm2() < 1e-10) continue;

        int n1 = network->segs[i].n1, n2 = network->segs[i].n2;
        bool c1 = (network->nodes[n1].constraint == INCLUSION_NODE);
        bool c2 = (network->nodes[n2].constraint == INCLUSION_NODE);
        if (!c1 && !c2) continue;

        int na = c1 ? n1 : n2, nb = c1 ? n2 : n1;

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
        Vec3 nrm = network->segs[i].plane.normalized();

        // ── 两端都是 c9(都在表面)→ 跨面弦 ──────────────────────────────
        // 这种弦在"面内穿出参数"判据里穿出点落在 nb 端(t≈1)被丢(nocross 漏)。
        // 改为直接判穿心+由两端的面定 corner:弦中点在夹杂内(确证穿心)、
        // 两端在不同轴的面 → 在共享棱 π∩E 插一个 c9,使段折回表面截线。
        if (c1 && c2) {
            Vec3 mid = network->cell.pbc_fold(C + 0.5*(l1 + l2));
            if (!is_node_in_inclusion(mid)) continue;   // 弦不穿内部(同面/沿棱)→ 无需插
            p_c9chord++;
            int ain = 0;
            if (fabs(l1[1]) > fabs(l1[ain])) ain = 1;
            if (fabs(l1[2]) > fabs(l1[ain])) ain = 2;
            int aout = 0;
            if (fabs(l2[1]) > fabs(l2[aout])) aout = 1;
            if (fabs(l2[2]) > fabs(l2[aout])) aout = 2;
            if (ain == aout) continue;   // 同轴(对面横穿)→ 单 corner 不适用,留待后续
            int afree = 3 - ain - aout;
            double d  = dot(nrm, l1);
            double si = (l1[ain]  >= 0.0) ? half : -half;
            double so = (l2[aout] >= 0.0) ? half : -half;
            Vec3 corner(0.0);
            corner[ain]  = si;
            corner[aout] = so;
            if (fabs(nrm[afree]) > 1e-9)
                corner[afree] = (d - nrm[ain]*si - nrm[aout]*so) / nrm[afree];
            else
                corner[afree] = 0.5*(l1[afree] + l2[afree]);
            if (corner[afree] >  half) corner[afree] =  half;
            if (corner[afree] < -half) corner[afree] = -half;
            // 顶点保护:三轴均贴 ±half → 夹到立方体顶点(脱滑移面),跳过(对齐一端自由路径)
            int xc3 = (fabs(corner[0]) >= half-2.0) + (fabs(corner[1]) >= half-2.0)
                    + (fabs(corner[2]) >= half-2.0);
            if (xc3 >= 3) continue;
            Vec3 pos = network->cell.pbc_fold(C + corner);
            if ((pos - network->nodes[na].pos).norm() < 1.0) continue;  // 退化:接触点≈端点
            if ((pos - network->nodes[nb].pos).norm() < 1.0) continue;
            int nnew = network->split_seg(i, pos);
            if (nnew < 0) continue;
            network->nodes[nnew].constraint = INCLUSION_NODE;
            network->nodes[nnew].v = Vec3(0.0);
            updated = true;
            p_c9corner++;
            continue;
        }

        // ── 一端 c9、一端自由 ────────────────────────────────────────────
        // 自由端 nb 不在面上,"面内穿出点"参数 0<t<1 成立,原判据本就能正确插角。
        if (inclusion_on_edge(network->nodes[na].pos, 2.0)) { p_anchor_edge++; continue; }

        Vec3 xcut;
        if (!inclusion_segment_edge_cross(l1, l2, nrm, half, xcut)) { p_nocross++; continue; }
        // 穿出点夹到立方体顶点(三轴均±half):退化,跳过
        int xc_corner = (fabs(xcut.x) >= half-2.0) + (fabs(xcut.y) >= half-2.0)
                      + (fabs(xcut.z) >= half-2.0);
        if (xc_corner >= 3) { p_vertex++; continue; }

        Vec3 pos = network->cell.pbc_fold(C + xcut);

        if ((pos - network->nodes[na].pos).norm() < min_sep) { p_minsep++; continue; }
        if ((pos - network->nodes[nb].pos).norm() < min_sep) { p_minsep++; continue; }

        int nnew = network->split_seg(i, pos);
        if (nnew < 0) continue;
        network->nodes[nnew].constraint = INCLUSION_NODE;
        network->nodes[nnew].v = Vec3(0.0);
        updated = true;
        p_insert++;
    }

    if (p_anchor_edge+p_nocross+p_vertex+p_minsep+p_insert+p_c9chord+p_c9corner > 0)
        ExaDiS_log("[EDGEPROBE] c9corner=%d/%d(chord) free_insert=%d skip: "
                   "vertex=%d minsep=%d anchoredge=%d nocross=%d\n",
                   p_c9corner, p_c9chord, p_insert, p_vertex, p_minsep, p_anchor_edge, p_nocross);

    if (updated) {
        network->generate_connectivity();
        network->update_ptr();
    }
}

} // namespace ExaDiS
