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
#include "inclusion.h"

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

    InclusionManager inclusion;

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
    Vec3 inclusion_surface_normal(const Vec3& pos) const;
    bool inclusion_on_edge(const Vec3& pos, double edge_tol) const;
    int inclusion_single_face_normal(const Vec3& pos, Vec3& normal) const;
   // void apply_inclusion_constraints();  // 检测并固定夹杂内的节点
    bool is_node_in_inclusion(const Vec3& pos) const;
    void update_inclusion_constraints(SerialDisNet* network); // 更新固定节点约束并将速度置零
   // void apply_inclusion_constraints(SerialDisNet* network);
    // �?system.h �?System 类声明中添加�?
    void insert_surface_nodes(SerialDisNet* network);
    void detect_orowan_loop(SerialDisNet* network);
    //void snap_surface_nodes_to_inclusion(SerialDisNet* network);
    //bool first_surface_insertion = true;
    bool is_node_strictly_inside_inclusion(const Vec3& pos) const;
    void project_surface_node_velocity(SerialDisNet* network);
    void correct_surface_node_positions(SerialDisNet* network);
    void insert_edge_nodes(SerialDisNet* network);
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
