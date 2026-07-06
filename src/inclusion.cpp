/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Inclusion interaction manager
 *
 *-------------------------------------------------------------------------*/

#include "system.h"
#include "inclusion_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace ExaDiS {
void InclusionManager::initialize(System* system, SerialDisNet *network)
{
    double inclusion_a_phys = 0.0;
    const char* a_str = std::getenv("INCLUSION_A");
    if (a_str != nullptr) {
        inclusion_a_phys = std::stod(a_str);
        if (inclusion_a_phys > 0.0) enabled = true;
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

    if (enabled && !initialized) {
        a_dim = inclusion_a_phys / system->params.burgmag;

        double Lx_dim = network->cell.H.xx();
        double Ly_dim = network->cell.H.yy();
        double Lz_dim = network->cell.H.zz();

        double box_vol = Lx_dim * Ly_dim * Lz_dim;
        double inclusion_vol = a_dim * a_dim * a_dim;
        int N_total = (int)round(vol_frac * box_vol / inclusion_vol);
        int Nx = (int)round(pow((double)N_total, 1.0/3.0));
        if (Nx < 1) Nx = 1;
        int Ny = Nx, Nz = Nx;

        double dx = Lx_dim / Nx;
        double dy = Ly_dim / Ny;
        double dz = Lz_dim / Nz;

        if (a_dim > dx || a_dim > dy || a_dim > dz) {
            ExaDiS_log("Warning: inclusion size may cause overlap! a_dim=%.2f, dx=%.2f, dy=%.2f, dz=%.2f\n",
                       a_dim, dx, dy, dz);
        }

        centers.clear();
        for (int i = 0; i < Nx; ++i) {
            double cx = dx * (i + 0.5);
            for (int j = 0; j < Ny; ++j) {
                double cy = dy * (j + 0.5);
                for (int k = 0; k < Nz; ++k) {
                    double cz = dz * (k + 0.5);
                    centers.push_back(Vec3(cx, cy, cz));
                }
            }
        }
        centers_valid = true;

        ExaDiS_log("Inclusion array: %d x %d x %d = %d inclusions, a_dim=%.2f, vol_frac=%.2f\n",
                   Nx, Ny, Nz, (int)centers.size(), a_dim, vol_frac);
        for (size_t idx = 0; idx < std::min((size_t)10, centers.size()); ++idx) {
            ExaDiS_log("  Inclusion %zu: center=(%.2f, %.2f, %.2f)\n",
                       idx, centers[idx].x, centers[idx].y, centers[idx].z);
        }

        initialized = true;
    }
}

void InclusionManager::before_integrate(System* system, SerialDisNet* network)
{
    if (!enabled) return;
    project_surface_node_velocity(system, network);
}

void InclusionManager::after_integrate(System* system, SerialDisNet* network)
{
    if (!enabled) return;
    insert_surface_nodes(system, network);
    insert_edge_nodes(system, network);
    update_constraints(system, network);
    correct_surface_node_positions(system, network);
}

void InclusionManager::after_reset_glide(System* system, SerialDisNet* network)
{
    if (!enabled) return;
    update_constraints(system, network);
}

void InclusionManager::after_collision(System* system, SerialDisNet* network)
{
    if (!enabled) return;
    detect_orowan_loop(network);
}

void InclusionManager::after_topology_remesh(System* system, SerialDisNet* network)
{
    if (!enabled) return;
    insert_surface_nodes(system, network);
    insert_edge_nodes(system, network);
    update_constraints(system, network);
    correct_surface_node_positions(system, network);
}

bool InclusionManager::is_node_in_inclusion(const Vec3& pos) const {
    if (!enabled) return false;
    double half = a_dim * 0.5;
    double tol = 1e-6 * a_dim;
    for (const auto& center : centers) {
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
 *    Function:     InclusionManager::is_node_strictly_inside_inclusion()
 *
 *    銆愪慨澶?銆戝宸粠 1e-3*a_dim 鏀逛负 0.05*a_dim
 *    纭繚琛ㄩ潰 PINNED 鑺傜偣锛堝潗鏍囪宸害 1~10b锛変笉琚鍒や负鍐呴儴鑺傜偣銆?
 *    0.05*a_dim = 114b锛岃繙澶т簬鍧愭爣璇樊锛岃繙灏忎簬澶规潅鍗婂緞锛?137b锛夈€?
 *
 *-------------------------------------------------------------------------*/
bool InclusionManager::is_node_strictly_inside_inclusion(const Vec3& pos) const {
    if (!enabled) return false;
    double half = a_dim * 0.5;
    // 瀹瑰樊浠?1e-3 鏀逛负 0.05锛氳〃闈㈣妭鐐瑰潗鏍囧湪 卤half 闄勮繎锛?
    // 鍐呯缉 0.05*a_dim 鍚庤〃闈㈣妭鐐逛笉浼氳鍒や负鍐呴儴鑺傜偣
    double tol = 0.05 * a_dim;
    for (const auto& center : centers) {
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

void InclusionManager::update_constraints(System* system, SerialDisNet* network) {
    if (!enabled) return;
    if (network == nullptr) return;

    int nnodes = network->number_of_nodes();
    static int total_fixed_count = 0;
    int new_fixed = 0;
    int new_projected = 0;
    double half = a_dim * 0.5;

    const double CONTACT_MARGIN = 5.0;  // 鎺ヨЕ鍒ゅ畾浣欓噺(b):鎶撹创闈絾灏忓箙鍋忓嚭鐨勮妭鐐?

    // ============================================================
    // 绗竴姝ワ細鎵弿鑺傜偣锛屾崟鑾峰湪鍐呴儴銆佽〃闈笂銆佹垨琛ㄩ潰澶栨瀬杩戠殑鑺傜偣
    // ============================================================
    for (int i = 0; i < nnodes; ++i) {
        if (network->nodes[i].constraint == PINNED_NODE) continue;
        // 鏁戣€屼笉鍒?宸插湪琛ㄩ潰鐨?c9 鐓ф棫璺宠繃;婕傝繘澶规潅鍐呴儴鐨?c9 涓嶈烦杩?钀藉埌涓嬮潰鐢?
        // glide_line 褰掍綅鍥炶〃闈?淇?d),浠庤€屽惊鐜? 鎵埌鏃跺凡鏃?鍐呴儴绔偣",鍏舵涓嶈娓呴浂鍒犳帀銆?
        if (network->nodes[i].constraint == INCLUSION_NODE &&
            !is_node_strictly_inside_inclusion(network->nodes[i].pos)) continue;

        Vec3 pos = network->nodes[i].pos;

        // ---- 鍒ゅ畾鑺傜偣搴旇琚鐞嗙殑鍏蜂綋澶规潅鍜屾儏褰?----
        int  detected_incl  = -1;
        bool is_inside_or_on = false;

        // 绗竴閬嶏細鏌ユ壘"鍦ㄥ唴鎴栧湪琛ㄩ潰"鐨勫す鏉?
        for (int k = 0; k < (int)centers.size(); k++) {
            Vec3 local = pos - centers[k];
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

        // ---- 鎵惧埌鏈€杩戦潰銆佽绠楁姇褰变綅缃拰娉曞悜閲?----
        Vec3 c = centers[detected_incl];
        Vec3 local = pos - c;

        // 閫夋渶杩戝崟闈?
        int face_sign[3];
        inclusion_nearest_face(local, face_sign);

        // 鎹曡幏鍓嶈妭鐐逛粛鍦ㄧ湡瀹炴粦绉婚潰鍐?ExaDiS 婊戠Щ绾︽潫淇濊瘉),娌块潰娉曞悜纭憗浼氭妸瀹?
        // 韪㈢鐪熷钩闈?螖d = glide_n[闈㈣酱]脳绌块€忔繁搴?銆傛敼鐢ㄤ繚 d 鐨?glide_line 鎶曞奖,
        // 钀界偣涓ユ牸鍦?(闈?鈭?婊戠Щ闈? 浜ょ嚎涓?涓?correct_surface_node_positions 涓€鑷淬€?
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
                       : inclusion_project_capture(local, face_sign, half, 30.0); // 鏃犳粦绉婚潰鏃堕€€鍥?

        // ---- 鎶曞奖 + PIN ----
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
    // 绗簩姝ワ細娓呴櫎骞界伒娈碉紙淇濇寔鍘熸湁閫昏緫浣滀负鍏滃簳锛?
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
        ExaDiS_log("Step: new fixed=%d (projected=%d), total fixed=%d, ghost segs zeroed=%d\n",
                   new_fixed, new_projected, total_fixed_count, ghost_segs);
        if (ghost_segs > 0) {
            ExaDiS_log("*** Orowan ring candidate: ghost segs = %d ***\n", ghost_segs);
        }
    }
}

/*---------------------------------------------------------------------------
 *
 *    Function:     InclusionManager::seg_cube_intersect()
 *
 *    璁＄畻绾挎 [p_out 鈫?p_in] 涓庤酱瀵归綈绔嬫柟浣撶殑鍏ュ皠闈氦鐐广€?
 *
 *    淇锛?
 *      鍒犻櫎鍘熸潵鏈夌己闄风殑"妫辫鍚搁檮"閫昏緫銆?
 *      鍘熼€昏緫鍦ㄦ１绾块檮杩戜細鎶婅妭鐐规帹鍒版１绾夸笂锛屽鑷磋妭鐐瑰亸绂荤湡瀹炰氦鐐广€?
 *      淇鍚庢敼涓?鏈€杩戦潰鎶曞奖"锛氭壘鍒颁氦鐐规渶闈犺繎鐨勯偅涓潰锛?
 *      鍙妸璇ユ柟鍚戠殑鍧愭爣绮剧‘璁句负 卤half锛屽叾浣欏潗鏍囦繚鎸?slab 璁＄畻缁撴灉銆?
 *
 *-------------------------------------------------------------------------*/
bool InclusionManager::seg_cube_intersect(const Vec3& p_out, const Vec3& p_in,
                                 const Vec3& center, double half, Vec3& hit) const
{
    Vec3 d = p_in - p_out;
    double tmin = 0.0, tmax = 1.0;
 
    for (int k = 0; k < 3; ++k) {
        double lo = center[k] - half;
        double hi = center[k] + half;
        if (fabs(d[k]) < 1e-14) {
            // 璇ユ柟鍚戝钩琛岋細妫€鏌ユ槸鍚﹀湪 slab 鍐?
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
 
    // tmin 瀵瑰簲浠?p_out 鍑哄彂绗竴涓繘鍏ュす鏉傜殑闈紙鍏ュ皠闈氦鐐癸級
    hit = p_out + tmin * d;
 
    // 鏈€杩戦潰鎶曞奖锛氭秷闄ゆ诞鐐硅宸紝鎶?hit 绮剧‘鏀惧埌鏈€杩戠殑闈笂
    // 鍙慨姝ｆ渶杩戦潰鏂瑰悜鐨勫潗鏍囷紝鍏朵綑鍧愭爣淇濇寔涓嶅彉
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
 *    Function:     InclusionManager::seg_cube_clip()
 *    涓ょ鍦ㄥ銆佸鸡妯垏澶规潅(瑙?妫辨枩鍏ュ皠)鏃?杩斿洖鍒囧叆鐐?hit_in 鍜屽垏鍑虹偣 hit_out銆?
 *    瑕佹眰 0 < tmin < tmax < 1(涓ょ涓ユ牸鍦ㄥ)銆?
 *-------------------------------------------------------------------------*/
bool InclusionManager::seg_cube_clip(const Vec3& pa, const Vec3& pb,
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

void InclusionManager::project_surface_node_velocity(System* system, SerialDisNet* network)
{
    if (!enabled) return;

    int nnodes = network->number_of_nodes();
    for (int i = 0; i < nnodes; i++) {
        if (network->nodes[i].constraint != INCLUSION_NODE) continue;
        if (on_edge(network->nodes[i].pos, 2.0)) {
            network->nodes[i].v = Vec3(0.0);
            continue;
        }
        Vec3 n_surface;
        if (single_face_normal(network->nodes[i].pos, n_surface) < 0) continue;

        // Same connected non-ghost segment plane that correct_surface_node_positions
        // uses 鈫?velocity stays on the SAME 1D (face 鈭?glide-plane) line as the position.
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
 *    銆愪慨澶?銆戠浜岄樁娈垫敼涓哄鐞嗕换鎰忔暟閲忕殑澶栦晶閭诲眳锛堜笉鍐嶅彧澶勭悊 size==2锛?
 *    銆愪慨澶?銆慳dd_seg 鍓嶆鏌ユ闀匡紝闃叉 PBC 瀵艰嚧鐨勮秴闀挎
 *
 *-------------------------------------------------------------------------*/
void InclusionManager::insert_surface_nodes(System* system, SerialDisNet* network)
{
    if (!enabled) return;
 
    double half = a_dim * 0.5;
 
    // ================================================================
    // 绗竴闃舵锛氭壘鍒版墍鏈夎法瓒婂す鏉傝竟鐣岀殑绾挎锛屽湪浜ょ偣澶勬彃鍏?PINNED 鑺傜偣
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

        if (p1_in && p2_in) continue;  // 鍏ㄥ湪鍐呴儴,浜ょ粰绗簩闃舵鍒犻櫎

        if (!p1_in && !p2_in) {
            // 鍒囧叆:涓ょ鍦ㄥ銆佸鸡妯垏澶规潅(瑙?妫辨枩鍏ュ皠)
            for (int incl_idx = 0; incl_idx < (int)centers.size(); incl_idx++) {
                Vec3 hin, hout;
                if (seg_cube_clip(p1, p2, centers[incl_idx], half, hin, hout)) {
                    to_clip.push_back({i, hin, hout, incl_idx});
                    break;
                }
            }
            continue;
        }
        // 浠ヤ笅:涓€绔湪鍐呬竴绔湪澶?鍘熸湁鍗曠偣閫昏緫)

        int inner_node  = p1_in ? n1 : n2;
        Vec3 p_in      = p1_in ? p1 : p2;
        Vec3 p_out_raw  = p1_in ? p2 : p1;
 
        // 澧為噺寮忚繃婊?
        long long inner_key = network->nodes[inner_node].tag.domain * 1000000LL
                            + network->nodes[inner_node].tag.index;
        auto it = node_was_inside.find(inner_key);
        if (it != node_was_inside.end() && it->second) continue;
 
        bool n1_in = p1_in, n2_in = p2_in;
        if ((n1_pinned && n2_in) || (n2_pinned && n1_in)) continue;
 
        // 鎵?p_in 鎵€灞炲す鏉傦紝浠?p_in 涓哄熀鍑嗘姌鍙?p_out
        Vec3 best_hit;
        bool found = false;
        int found_incl_id = -1;
        for (int incl_idx = 0; incl_idx < (int)centers.size(); incl_idx++) {
            const Vec3& center = centers[incl_idx];
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
 
        double tol2 = (0.05 * a_dim) * (0.05 * a_dim);
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

    // 鍒囧叆鍙岀偣鎹曡幏:涓ょ鍦ㄥ銆佸鸡妯垏澶规潅,鍦ㄥ垏鍏ョ偣鍜屽垏鍑虹偣鍚勬彃涓€涓?c9 鑺傜偣
    std::sort(to_clip.begin(), to_clip.end(),
              [](const auto& a, const auto& b){ return a.seg_id > b.seg_id; });
    int clip_corner = 0;  // [EDGEPROBE] clip 娈典笂鎻掑叆鐨勬１鎺ヨЕ鑺傜偣鏁?
    for (auto& info : to_clip) {
        int seg_id = info.seg_id;
        if (seg_id >= network->number_of_segs()) continue;
        int cn1 = network->segs[seg_id].n1;
        int cn2 = network->segs[seg_id].n2;
        if (network->nodes[cn1].constraint == INCLUSION_NODE &&
            network->nodes[cn2].constraint == INCLUSION_NODE) continue;

        int nodeIn = network->split_seg(seg_id, info.hit_in);
        if (nodeIn < 0) continue;
        int s2 = network->number_of_segs() - 1;  // split_seg 杩藉姞杩滅鍗婃鍦ㄦ湯灏?
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

            // 鈹€鈹€ 褰撳満娑堥櫎绌垮績鎴?鍦ㄥ垏鍏?鍒囧嚭鐐逛箣闂存彃妫辨帴瑙﹁妭鐐?鈹€鈹€
            // 鍒囧叆闈笌鍒囧嚭闈㈠叡浜竴鏉＄珛鏂逛綋妫?E;鎺ヨЕ鐐?= 娈垫粦绉婚潰 蟺 鈭?E銆?
            // s2 姝ゅ埢鏄?nodeIn鈫抧odeOut 杩欐潯绌垮績鎴?鎶婂畠鍦?corner 澶勬姌寮€,
            // 浣?nodeIn鈫抍orner銆乧orner鈫抧odeOut 鍚勮嚜璐村湪涓€涓潰涓娿€佷笉杩涘唴閮ㄣ€?
            Vec3 C = centers[info.incl_id];
            Vec3 lin  = info.hit_in  - C;
            Vec3 lout = info.hit_out - C;
            int ain = 0;
            if (fabs(lin[1])  > fabs(lin[ain]))  ain  = 1;
            if (fabs(lin[2])  > fabs(lin[ain]))  ain  = 2;
            int aout = 0;
            if (fabs(lout[1]) > fabs(lout[aout])) aout = 1;
            if (fabs(lout[2]) > fabs(lout[aout])) aout = 2;
            Vec3 npl = network->segs[s2].plane;
            // 浠呯浉閭婚潰(涓嶅悓杞?鎵嶆湁鍏变韩妫卞彲缁?鍚岄潰/瀵归潰(鍚岃酱)璺宠繃
            if (ain != aout && npl.norm2() > 1e-10) {
                Vec3 nrm = npl.normalized();
                int afree = 3 - ain - aout;
                double d  = dot(nrm, lin);              // lin 鍦ㄦ粦绉婚潰 蟺 涓?
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
                // 閫€鍖栦繚鎶?鎺ヨЕ鐐逛笌鏌愮鐐瑰嚑涔庨噸鍚堝垯涓嶆彃(閬垮厤闆堕暱娈?
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
    // 绗簩闃舵锛氳惤鍏ュす鏉傚唴閮ㄧ殑鑺傜偣 鈫?鎺ㄥ洖琛ㄩ潰銆佹爣 c9锛堜笉鍒犺妭鐐广€佷笉閲嶈繛锛夈€?
    //   鎶曞奖澶嶇敤 correct_surface_node_positions 鐨?glide 鎰熺煡妫?瑙掓姇褰憋紝
    //   杩為€氭€у畬鍏ㄤ笉鍔紝浠庢牴涓婃潨缁濇棫閫昏緫鐨勬壇鏂?閿欏苟/瀛ょ珛纰庣墖銆?
    //   缁曟１鐨?鍏堝湪娈典笂鐢熸垚鑺傜偣鍐嶆斁鍒?蟺鈭╂１"鐢变繚鐣欑殑 insert_edge_nodes 璐熻矗銆?
    // ================================================================
    for (int i = 0; i < network->number_of_nodes(); i++) {
        if (network->nodes[i].constraint == INCLUSION_NODE) continue;  // 宸叉槸 c9
        if (network->nodes[i].constraint == PINNED_NODE)    continue;  // 閽夋墡婧愮涓嶅姩
        if (!is_node_strictly_inside_inclusion(network->nodes[i].pos)) continue;
        network->nodes[i].constraint = INCLUSION_NODE;
        network->nodes[i].v = Vec3(0.0);
    }
    correct_surface_node_positions(system, network);   // 灏卞湴娌?glide 鎶婃柊鏍?c9 鎶曞奖鍒拌〃闈?
 
    network->generate_connectivity();
    network->update_ptr();
 
    // 鏇存柊 node_was_inside 鐘舵€佽褰?
    node_was_inside.clear();
    for (int i = 0; i < network->number_of_nodes(); i++) {
        if (is_node_in_inclusion(network->nodes[i].pos)) {
            long long key = network->nodes[i].tag.domain * 1000000LL
                          + network->nodes[i].tag.index;
            node_was_inside[key] = true;
        }
    }
}

void InclusionManager::detect_orowan_loop(SerialDisNet* network)
{
    if (!enabled) return;

    for (int i = 0; i < network->number_of_segs(); i++) {
        if (network->segs[i].burg.norm2() < 1e-20) {
            int n1 = network->segs[i].n1;
            int n2 = network->segs[i].n2;
            if (network->nodes[n1].constraint == INCLUSION_NODE &&
                network->nodes[n2].constraint == INCLUSION_NODE) {
                orowan_loop_count++;
                ExaDiS_log("Orowan loop detected! Total count: %d\n", orowan_loop_count);
                return;  // 姣忔鏈€澶氳褰曚竴娆?
            }
        }
    }
}
/*---------------------------------------------------------------------------
 *
 *    Function:     System::correct_surface_node_positions()
 *
 *    姣忔 integrate 涔嬪悗璋冪敤銆傛壂鎻忔墍鏈夎〃闈㈣妭鐐癸紝鎶婂畠浠殑浣嶇疆鏍℃鍥?
 *    涓ユ牸鐨勮〃闈綅缃紝鍘婚櫎鏁板€兼紓绉汇€?
 *
 *    鏍℃瑙勫垯锛?
 *    - 娉曞悜閲?n 鐨勬煇涓垎閲?|n[k]| > 0.5锛氬己鍒?local[k] = sign(n[k]) * half
 *    - 娉曞悜閲?n 鐨勬煇涓垎閲?|n[k]| 鈮?0.5锛氳杞磋嚜鐢憋紙淇濈暀 local[k] 鍘熷€硷級
 *
 *-------------------------------------------------------------------------*/
void InclusionManager::correct_surface_node_positions(System* system, SerialDisNet* network)
{
    if (!enabled) return;
    if (network == nullptr) return;
    if (centers.empty()) return;

    double half = a_dim * 0.5;

    int nnodes = network->number_of_nodes();
    for (int i = 0; i < nnodes; i++) {
        if (network->nodes[i].constraint != INCLUSION_NODE) continue;
        int incl_id = -1; double bestd2 = 1e30;
        for (int k = 0; k < (int)centers.size(); k++) {
            Vec3 d = network->nodes[i].pos - centers[k];
            double dd = dot(d, d);
            if (dd < bestd2) { bestd2 = dd; incl_id = k; }
        }
        if (incl_id < 0) continue;
        Vec3 center = centers[incl_id];

        // Realtime classify: face assignment from CURRENT position, never stale stored normal.
        Vec3 old_pos = network->nodes[i].pos;
        Vec3 local = network->cell.pbc_position(center, old_pos) - center;
        int face_sign[3];
        inclusion_nearest_face(local, face_sign);

        // Stage 6: get this node's glide plane from a connected non-ghost segment,
        // then project position onto the (face 鈭?glide-plane) 1D line so the node
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
        // Sub-type by position: two axes near 卤half 鈫?edge/corner node, pin to 蟺鈭〦.
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
 *    c9 corner node at the 蟺鈭〦 intersection (glide plane 鈭?cube edge).
 *
 *-------------------------------------------------------------------------*/
void InclusionManager::insert_edge_nodes(System* system, SerialDisNet* network)
{
    if (!enabled || centers.empty()) return;
    double half = a_dim * 0.5;
    const double min_sep = 5.0;

    int nsegs = network->number_of_segs();
    bool updated = false;
    // [EDGEPROBE] 鍒嗘敮鍛戒腑璁℃暟(姣忔璋冪敤涓€琛?
    int p_anchor_edge=0, p_nocross=0, p_vertex=0, p_minsep=0, p_insert=0;
    int p_c9chord=0, p_c9corner=0;   // C9-C9 绌垮績寮?妫€鍑?/ 鎴愬姛鎻掕
    int p_c9_axiskip=0, p_c9_vertex=0, p_c9_degen=0;  // [EDGEPROBE] C9-C9 璺宠繃鍘熷洜(璇婃柇:鎹曟崏)

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
        for (int k = 0; k < (int)centers.size(); k++) {
            Vec3 dd = network->nodes[na].pos - centers[k];
            double q = dot(dd, dd);
            if (q < bestd) { bestd = q; incl = k; }
        }
        Vec3 C = centers[incl];
        Vec3 l1 = network->cell.pbc_position(C, network->nodes[na].pos) - C;
        Vec3 l2 = network->cell.pbc_position(C, network->nodes[nb].pos) - C;
        Vec3 nrm = network->segs[i].plane.normalized();

        // 鈹€鈹€ 涓ょ閮芥槸 c9(閮藉湪琛ㄩ潰)鈫?璺ㄩ潰寮?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        // 杩欑寮﹀湪"闈㈠唴绌垮嚭鍙傛暟"鍒ゆ嵁閲岀┛鍑虹偣钀藉湪 nb 绔?t鈮?)琚涪(nocross 婕?銆?
        // 鏀逛负鐩存帴鍒ょ┛蹇?鐢变袱绔殑闈㈠畾 corner:寮︿腑鐐瑰湪澶规潅鍐?纭瘉绌垮績)銆?
        // 涓ょ鍦ㄤ笉鍚岃酱鐨勯潰 鈫?鍦ㄥ叡浜１ 蟺鈭〦 鎻掍竴涓?c9,浣挎鎶樺洖琛ㄩ潰鎴嚎銆?
        if (c1 && c2) {
            Vec3 mid = network->cell.pbc_fold(C + 0.5*(l1 + l2));
            if (!is_node_in_inclusion(mid)) continue;   // 寮︿笉绌垮唴閮?鍚岄潰/娌挎１)鈫?鏃犻渶鎻?
            p_c9chord++;
            int ain = 0;
            if (fabs(l1[1]) > fabs(l1[ain])) ain = 1;
            if (fabs(l1[2]) > fabs(l1[ain])) ain = 2;
            int aout = 0;
            if (fabs(l2[1]) > fabs(l2[aout])) aout = 1;
            if (fabs(l2[2]) > fabs(l2[aout])) aout = 2;
            if (ain == aout) { p_c9_axiskip++; continue; }   // 鍚岃酱(瀵归潰妯┛)鈫?鍗?corner 涓嶉€傜敤,鐣欏緟鍚庣画
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
            // 椤剁偣淇濇姢:涓夎酱鍧囪创 卤half 鈫?澶瑰埌绔嬫柟浣撻《鐐?鑴辨粦绉婚潰),璺宠繃(瀵归綈涓€绔嚜鐢辫矾寰?
            int xc3 = (fabs(corner[0]) >= half-2.0) + (fabs(corner[1]) >= half-2.0)
                    + (fabs(corner[2]) >= half-2.0);
            if (xc3 >= 3) { p_c9_vertex++; continue; }
            Vec3 pos = network->cell.pbc_fold(C + corner);
            if ((pos - network->nodes[na].pos).norm() < min_sep) { p_c9_degen++; continue; }  // 瑙掔偣绂荤鐐?< min_sep:璺宠繃(瀵归綈鑷敱鍒嗘敮,鏂嚫妫遍綈璇虹粏鍒?
            if ((pos - network->nodes[nb].pos).norm() < min_sep) { p_c9_degen++; continue; }
            int nnew = network->split_seg(i, pos);
            if (nnew < 0) continue;
            network->nodes[nnew].constraint = INCLUSION_NODE;
            network->nodes[nnew].v = Vec3(0.0);
            updated = true;
            p_c9corner++;
            continue;
        }

        // 鈹€鈹€ 涓€绔?c9銆佷竴绔嚜鐢?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        // 鑷敱绔?nb 涓嶅湪闈笂,"闈㈠唴绌垮嚭鐐?鍙傛暟 0<t<1 鎴愮珛,鍘熷垽鎹湰灏辫兘姝ｇ‘鎻掕銆?
        if (on_edge(network->nodes[na].pos, 2.0)) { p_anchor_edge++; continue; }

        Vec3 xcut;
        if (!inclusion_segment_edge_cross(l1, l2, nrm, half, xcut)) { p_nocross++; continue; }
        // 绌垮嚭鐐瑰す鍒扮珛鏂逛綋椤剁偣(涓夎酱鍧嚶県alf):閫€鍖?璺宠繃
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
                   "vertex=%d minsep=%d anchoredge=%d nocross=%d | c9skip: axis=%d vert=%d degen=%d\n",
                   p_c9corner, p_c9chord, p_insert, p_vertex, p_minsep, p_anchor_edge, p_nocross,
                   p_c9_axiskip, p_c9_vertex, p_c9_degen);

    if (updated) {
        network->generate_connectivity();
        network->update_ptr();
    }
}

} // namespace ExaDiS

