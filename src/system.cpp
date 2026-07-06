/*---------------------------------------------------------------------------
 *
 *	ExaDiS
 *
 *	Nicolas Bertin
 *	bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include "system.h"

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
    params = _params;
    crystal = _crystal;

#ifdef MPI
    ExaDiS_fatal("System::initialize not implemented for MPI\n");
#endif

    if (!network)
        ExaDiS_fatal("Error: undefined initial dislocation configuration\n");

    net_mngr = make_network_manager(network);

    inclusion.initialize(this, network);
    if (inclusion.enabled) network->recycle = false;  // [VERIFY] disable index recycling to confirm stale-label inheritance root cause

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
 *    Function:     System inclusion compatibility wrappers
 *
 *-------------------------------------------------------------------------*/
Vec3 System::inclusion_surface_normal(const Vec3& pos) const
{
    return inclusion.surface_normal(pos);
}

bool System::inclusion_on_edge(const Vec3& pos, double edge_tol) const
{
    return inclusion.on_edge(pos, edge_tol);
}

int System::inclusion_single_face_normal(const Vec3& pos, Vec3& normal) const
{
    return inclusion.single_face_normal(pos, normal);
}

bool System::is_node_in_inclusion(const Vec3& pos) const
{
    return inclusion.is_node_in_inclusion(pos);
}

bool System::is_node_strictly_inside_inclusion(const Vec3& pos) const
{
    return inclusion.is_node_strictly_inside_inclusion(pos);
}

void System::update_inclusion_constraints(SerialDisNet* network)
{
    inclusion.update_constraints(this, network);
}

void System::project_surface_node_velocity(SerialDisNet* network)
{
    inclusion.project_surface_node_velocity(this, network);
}

void System::insert_surface_nodes(SerialDisNet* network)
{
    inclusion.insert_surface_nodes(this, network);
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
    inclusion.detect_orowan_loop(network);
}

void System::correct_surface_node_positions(SerialDisNet* network)
{
    inclusion.correct_surface_node_positions(this, network);
}

void System::insert_edge_nodes(SerialDisNet* network)
{
    inclusion.insert_edge_nodes(this, network);
}
} // namespace ExaDiS
