/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "types.h"
#include "crystal.h"
#include "params.h"
#include "oprec.h"
#include <vector>
#include <unordered_map>

#pragma once
#ifndef EXADIS_SYSTEM_H
#define EXADIS_SYSTEM_H

namespace ExaDiS {

extern FILE* flog;

class System {
public:
    DisNetManager* net_mngr = nullptr;
    double neighbor_cutoff;
    
    inline SerialDisNet* get_serial_network() { return net_mngr->get_serial_network(); }
    inline DeviceDisNet* get_device_network() { return net_mngr->get_device_network(); }
    
    inline int Nnodes_local() { return net_mngr->Nnodes_local(); }
    inline int Nsegs_local() { return net_mngr->Nsegs_local(); }
    
    inline int Nnodes_total() { return Nnodes_local(); }
    inline int Nsegs_total() { return Nsegs_local(); }
    
    T_x xold;
    
    Mat33 extstress;
    double realdt;
    Mat33 dEp, dWp;
    double density;

    bool inclusion_enabled = false;
    double inclusion_a_dim = 0.0;                     // 无量纲边长
    std::vector<Vec3> inclusion_centers;              // 所有夹杂中心
    bool inclusion_centers_valid = false;             // 是否已生成
    // 新增：计算夹杂表面法向量
    Vec3 inclusion_surface_normal(const Vec3& pos) const {
    double half = inclusion_a_dim * 0.5;
    double min_dist = 1e30;
    Vec3 normal(0.0, 0.0, 1.0);
    double tol = inclusion_a_dim * 0.05; // 5%边长作为棱角过渡区
    
    for (const auto& center : inclusion_centers) {
        Vec3 d = pos - center;
        double dx = fabs(fabs(d.x) - half);
        double dy = fabs(fabs(d.y) - half);
        double dz = fabs(fabs(d.z) - half);
        double dist = fmin(dx, fmin(dy, dz));
        
        if (dist < min_dist) {
            min_dist = dist;
            // 在棱角过渡区内，混合两个或三个面的法向量
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

    // True if pos is on a cube edge (two or more axes near ±half).
    bool inclusion_on_edge(const Vec3& pos, double edge_tol) const {
        if (inclusion_centers.empty()) return false;
        int best = 0; double bestd = 1e30;
        for (int k = 0; k < (int)inclusion_centers.size(); k++) {
            Vec3 d = pos - inclusion_centers[k];
            double dd = dot(d, d);
            if (dd < bestd) { bestd = dd; best = k; }
        }
        Vec3 l = pos - inclusion_centers[best];
        double half = inclusion_a_dim * 0.5;
        int near = (fabs(l.x) >= half-edge_tol) + (fabs(l.y) >= half-edge_tol)
                 + (fabs(l.z) >= half-edge_tol);
        return near >= 2;
    }

    // Flow-through edge velocity: node sits on a cube edge (>=2 axes near ±half).
    // Among the touched faces, pick the one the glide-plane velocity vi drives the
    // node INTO (every *other* pinned axis de-pins, moving toward the interior),
    // and return vi projected onto that face's slip-line l=cross(glide_n,n_face).
    // l ⊥ glide_n so motion preserves the slip-plane offset d. Returns false if no
    // face qualifies (true corner / pressed into inclusion) -> caller pins (vi=0).
    bool inclusion_edge_flow_velocity(const Vec3& pos, const Vec3& glide_n,
                                      const Vec3& vi, double edge_tol, Vec3& vout) const {
        if (inclusion_centers.empty()) return false;
        int best = 0; double bestd = 1e30;
        for (int k = 0; k < (int)inclusion_centers.size(); k++) {
            Vec3 d = pos - inclusion_centers[k];
            double dd = dot(d, d);
            if (dd < bestd) { bestd = dd; best = k; }
        }
        Vec3 local = pos - inclusion_centers[best];
        double half = inclusion_a_dim * 0.5;
        int pin[3]; int npin = 0;
        for (int k = 0; k < 3; k++)
            if (fabs(local[k]) >= half - edge_tol) pin[npin++] = k;
        if (npin < 2) return false;
        double best_speed = 0.0; bool found = false;
        for (int ii = 0; ii < npin; ii++) {
            int f = pin[ii];                          // face to stay on
            Vec3 n_f(0.0); n_f[f] = (local[f] >= 0.0) ? 1.0 : -1.0;
            Vec3 l = cross(glide_n, n_f);
            double ln = l.norm();
            if (ln < 1e-9) continue;
            l = (1.0/ln) * l;
            Vec3 v = dot(vi, l) * l;
            bool into = true;                         // every other pinned axis must de-pin
            for (int jj = 0; jj < npin; jj++) {
                if (jj == ii) continue;
                int g = pin[jj];
                double s = (local[g] >= 0.0) ? 1.0 : -1.0;
                if (v[g] * s >= 0.0) { into = false; break; }
            }
            if (!into) continue;
            double sp = v.norm();
            if (sp > best_speed) { best_speed = sp; vout = v; found = true; }
        }
        return found;
    }

    // 从位置现算最近单面法向量;取代 surface_node_normal 映射,保证僵尸节点也被约束
    int inclusion_single_face_normal(const Vec3& pos, Vec3& normal) const {
        normal = Vec3(0.0);
        if (inclusion_centers.empty()) return -1;
        int best = 0; double bestd = 1e30;
        for (int k = 0; k < (int)inclusion_centers.size(); k++) {
            Vec3 d = pos - inclusion_centers[k];
            double dd = dot(d, d);
            if (dd < bestd) { bestd = dd; best = k; }
        }
        Vec3 local = pos - inclusion_centers[best];
        double ax = fabs(local.x), ay = fabs(local.y), az = fabs(local.z);
        if (ax >= ay && ax >= az)
            normal = Vec3(local.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0);
        else if (ay >= ax && ay >= az)
            normal = Vec3(0.0, local.y >= 0.0 ? 1.0 : -1.0, 0.0);
        else
            normal = Vec3(0.0, 0.0, local.z >= 0.0 ? 1.0 : -1.0);
        return best;
    }

    Params params;
    Crystal crystal;
    
    System();
    ~System();
    System(const System&) = delete;
    void initialize(Params _params, Crystal _crystal, SerialDisNet* network);
    void register_neighbor_cutoff(double cutoff);
    void plastic_strain();
    void reset_glide_planes();
    void write_config(std::string filename);
   // void apply_inclusion_constraints();  // 检测并固定夹杂内的节点
    bool is_node_in_inclusion(const Vec3& pos) const;
    void update_inclusion_constraints(SerialDisNet* network); // 更新固定节点约束并将速度置零
   // void apply_inclusion_constraints(SerialDisNet* network);
    // 在 system.h 的 System 类声明中添加：
    void insert_surface_nodes(SerialDisNet* network);
    bool seg_cube_intersect(const Vec3& p_out, const Vec3& p_in,
                        const Vec3& center, double half, Vec3& hit) const;
    bool seg_cube_clip(const Vec3& pa, const Vec3& pb,
                       const Vec3& center, double half,
                       Vec3& hit_in, Vec3& hit_out) const;
    int orowan_loop_count = 0;
    void detect_orowan_loop(SerialDisNet* network);
    bool inclusion_initialized = false;
    bool same_inclusion_face(const Vec3& pos1, const Vec3& pos2) const;
    //void snap_surface_nodes_to_inclusion(SerialDisNet* network);
    //bool first_surface_insertion = true;
    bool is_node_strictly_inside_inclusion(const Vec3& pos) const;
    void snap_nodes_to_surface(SerialDisNet* network);
    Vec3 get_face_normal(const Vec3& hit, const Vec3& center, double half) const;
    void project_surface_node_velocity(SerialDisNet* network);
    void check_surface_node_transition(SerialDisNet* network);
    void enforce_edge_continuity(SerialDisNet* network);
    void correct_surface_node_positions(SerialDisNet* network);
    void insert_edge_nodes(SerialDisNet* network);
    // 记录上一步结束时在夹杂内部的节点 Tag ID
    // 用于增量式处理：只对本步新进入夹杂的节点插入表面节点
    std::unordered_map<long long, bool> node_was_inside;
    OpRec* oprec = nullptr;
    
    int num_ranks;
    int proc_rank;
    
    struct SystemTimer {
        Kokkos::Timer timer;
        double accumtime;
        std::string label;
        SystemTimer() { accumtime = 0.0; }
        SystemTimer(std::string _label) : label(_label) { accumtime = 0.0; }
        void start() { timer.reset(); }
        void stop() { accumtime += timer.seconds(); }
    };
    enum timers {TIMER_FORCE, TIMER_MOBILITY, TIMER_INTEGRATION, TIMER_CROSSSLIP,
                 TIMER_COLLISION, TIMER_TOPOLOGY, TIMER_REMESH, TIMER_OUTPUT, TIMER_END};
    SystemTimer timer[TIMER_END];
    
    bool pyexadis = false;
    static const int MAX_DEV_TIMERS = 20;
    int numdevtimer = 0;
    SystemTimer devtimer[MAX_DEV_TIMERS];
    int add_timer(std::string label) {
        if (pyexadis) return 0;
        if (numdevtimer == MAX_DEV_TIMERS)
            ExaDiS_fatal("Error: MAX_DEV_TIMERS = %d limit reached\n", MAX_DEV_TIMERS);
        devtimer[numdevtimer++].label = label;
        return numdevtimer-1;
    }
    void print_timers(double timetot=-1.0, bool dev=false);
};

System* make_system(SerialDisNet* net, Crystal crystal=Crystal(), Params params=Params());
DisNetManager* make_network_manager(SerialDisNet* net);

} // namespace ExaDiS

#endif
