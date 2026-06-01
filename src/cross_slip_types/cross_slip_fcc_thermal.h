/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Thermally-activated cross-slip for FCC crystals.
 *  Reference: Hussein, Ahmed M., et al. Acta Materialia 85 (2015): 180-190.
 *
 *  The algorithm groups connected screw segments into "chains" and evaluates
 *  thermodynamic transition probabilities driven by Escaig stress differences
 *  between the glide and cross-slip planes. Two chain types are supported:
 *    - Bulk:         chains fully inside the simulation volume
 *    - Intersection: chains terminating at dislocation junctions (Hirth,
 *                    Glide lock, LC lock)
 *
 *  Thermal probability:
 *    P = nu * dt * (L / L_ref) * exp(-(E_a - V_a * dSigma_Escaig) / (kB * T))
 *
 *
 *-------------------------------------------------------------------------*/

#pragma once
#ifndef EXADIS_CROSS_SLIP_FCC_THERMAL_H
#define EXADIS_CROSS_SLIP_FCC_THERMAL_H

#include <random>
#include <cmath>
#include <vector>
#include "force.h"
#include "cross_slip.h"

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Class:    CrossSlipFCCThermal
 *              Thermally-activated FCC cross-slip (serial implementation).
 *
 *-------------------------------------------------------------------------*/
class CrossSlipFCCThermal : public CrossSlip {
public:

    /*-----------------------------------------------------------------------
     *    Struct:   Params
     *              Material and cross-slip parameters. All activation
     *              energies are in Joules, stacking fault energies in J/m^2,
     *              lengths in units of burgmag.
     *---------------------------------------------------------------------*/
    struct Params {
        double temperature   = 300.0; ///< Temperature [K]
        int    evalFrequency = 1;     ///< Steps between evaluations

        // --- Bulk cross-slip ---
        double bulkActivationEnergy       = 0.8*1.602E-19;  ///< E_a [J]
        double bulkActivationVolume       = 20.0;           ///< V_a factor (need to multiple b^3)
        double bulkAttemptFrequency       = 5.0E17;         ///< nu [1/s]
        double bulkReferenceLength        = 1000.0;         ///< L_ref [b]

        // --- Hirth lock (intersection) ---
        double hirthActivationEnergy       = 0.2*1.602E-19;
        double hirthActivationVolume       = 20.0;
        double hirthAttemptFrequency       = 5.0E17;
        double hirthReferenceLength        = 1000.0;
        double hirthEffectiveLength        = 2.5e-9;        ///< 2.5nm (need to devide burgmag when use)

        // --- Glide lock (intersection) ---
        double glideLockActivationEnergy       = 0.5*1.602E-19;
        double glideLockActivationVolume       = 20.0;
        double glideLockAttemptFrequency       = 5.0E17;
        double glideLockReferenceLength        = 1000.0;
        double glideLockEffectiveLength        = 2.5e-9;    ///< 2.5nm (need to devide burgmag when use)

        // --- LC lock (intersection) ---
        double lcLockActivationEnergy       = 0.6*1.602E-19;
        double lcLockActivationVolume       = 20.0;
        double lcLockAttemptFrequency       = 5.0E17;
        double lcLockReferenceLength        = 1000.0;
        double lcLockEffectiveLength        = 2.5e-9;       ///< 2.5nm (need to devide burgmag when use)

        /// Segment is considered screw if its angle with b is within this threshold [deg]
        double screwAngleTolerance = 15.0;

        Params() = default;
    };

private:
    Force*  force;
    Params  params;
    int     eval_counter = 0;

    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform01{0.0, 1.0};

    enum MechanismType { Bulk, Intersection };
    enum LockType  { Hirth, GlideLock, LCLock, UnknownLock };

    /*-----------------------------------------------------------------------
     *    Struct:   ScrewChain
     *              Connected screw-character segments sharing the same
     *              Burgers vector and {111} glide plane.
     *---------------------------------------------------------------------*/
    struct ScrewChain {
        std::vector<int> node_ids;
        std::vector<int> seg_ids;
        Vec3 burg;
        Vec3 glide_plane;
        MechanismType mechanism = Bulk;
        LockType  junction  = UnknownLock;
        Vec3 junction_burg;
        Vec3 junction_plane;
    };

    // -----------------------------------------------------------------------
    //  Classify a vector by FCC crystallographic family
    // -----------------------------------------------------------------------
    static bool is_100_family(const Vec3& v) {
        const double tol = 0.1;
        Vec3 u = v.normalized();
        int nonzero = (fabs(u.x) > tol) + (fabs(u.y) > tol) + (fabs(u.z) > tol);
        return nonzero == 1;
    }

    static bool is_110_family(const Vec3& v) {
        const double tol = 0.1;
        Vec3 u = v.normalized();
        double ax = fabs(u.x), ay = fabs(u.y), az = fabs(u.z);
        if (ax < tol && fabs(ay - az) < tol && ay > tol) return true;
        if (ay < tol && fabs(ax - az) < tol && ax > tol) return true;
        if (az < tol && fabs(ax - ay) < tol && ax > tol) return true;
        return false;
    }

    static bool is_111_family(const Vec3& v) {
        const double tol = 0.1;
        Vec3 u = v.normalized();
        double ax = fabs(u.x), ay = fabs(u.y), az = fabs(u.z);
        return (fabs(ax - ay) < tol && fabs(ay - az) < tol && ax > tol);
    }

    static bool is_112_family(const Vec3& v) {
        const double tol = 0.1;
        Vec3 u = v.normalized();
        double ax = fabs(u.x), ay = fabs(u.y), az = fabs(u.z);
        double vals[3] = {ax, ay, az};
        double vmax = (ax > ay ? ax : ay);
        vmax = (vmax > az ? vmax : az);
        if (vmax < tol) return false;
        int n_large = 0, n_small = 0;
        for (int i = 0; i < 3; i++) {
            if      (fabs(vals[i] - vmax) < tol)          n_large++;
            else if (fabs(2.0 * vals[i] - vmax) < tol)    n_small++;
            else if (vals[i] < tol) {}
            else return false;
        }
        return (n_large == 1 && n_small == 2);
    }

    // -----------------------------------------------------------------------
    //  Returns the cross-slip plane normal for an FCC 1/2<110> dislocation.
    //  Given glide plane n and Burgers vector b, returns the other {111} plane
    //  containing b by flipping the sign of the zero component of b in n.
    // -----------------------------------------------------------------------
    static Vec3 get_crossslip_plane(const Vec3& glide_plane_normal,
                                    const Vec3& burg) {
        const double tol = 1e-6;
        Vec3 bn = burg.normalized();
        if (fabs(dot(bn, glide_plane_normal)) > tol) return Vec3(0.0);
        if (fabs(bn.x) < tol) return Vec3(-glide_plane_normal.x,  glide_plane_normal.y,  glide_plane_normal.z);
        if (fabs(bn.y) < tol) return Vec3( glide_plane_normal.x, -glide_plane_normal.y,  glide_plane_normal.z);
        if (fabs(bn.z) < tol) return Vec3( glide_plane_normal.x,  glide_plane_normal.y, -glide_plane_normal.z);
        return Vec3(0.0);
    }

    // -----------------------------------------------------------------------
    //  Escaig stress component from accumulated chain force.
    // -----------------------------------------------------------------------
    static double compute_escaig_stress(const Vec3& force,
                                        const Vec3& burg_dir,
                                        const Vec3& plane_normal,
                                        double      burg_mag,
                                        double      total_length) {
        if (burg_mag < 1e-15 || total_length < 1e-15) return 0.0;
        Vec3 escaig_dir = cross(plane_normal, burg_dir);
        double len = escaig_dir.norm();
        if (len < 1e-10) return 0.0;
        return dot(force, escaig_dir / len) / (burg_mag * total_length);
    }

    // -----------------------------------------------------------------------
    //  Arrhenius cross-slip probability per time step (Hussein et al. Eq. 2):
    //  P = nu * dt * (L / L_ref) * exp(-(E_a - V_a * dSigma_E) / (kB * T))
    // -----------------------------------------------------------------------
    static double compute_thermal_probability(double E_a, double V_a,
                                              double dSigma_E, double temperature,
                                              double attempt_frequency, double dt,
                                              double chain_length, double reference_length) {
        const double kB = 1.3806503e-23;
        double exponent = -(E_a - V_a * dSigma_E) / (kB * temperature);
        return attempt_frequency * dt * (chain_length / reference_length) * exp(exponent);
    }

    // -----------------------------------------------------------------------
    //  Builds all screw chain candidates from physical dislocation links.
    //  Applies FCC crystallographic filters and the screw angle criterion.
    // -----------------------------------------------------------------------
    std::vector<ScrewChain> build_screw_chains(System* system,
                                               SerialDisNet* network)
    {
        const double tol  = 1e-6;
        const double scos = cos(params.screwAngleTolerance * M_PI / 180.0);

        SerialDisNet::DisLinks dislinks = network->physical_links();
        std::vector<ScrewChain> chains;

        for (int l = 0; l < dislinks.number_of_links; l++) {
            const auto& snodes = dislinks.links_nodes[l];
            const auto& ssegs  = dislinks.links_segs[l];
            if (ssegs.empty()) continue;
            int nseg = (int)ssegs.size();

            // Burgers vector, sign-corrected for traversal direction
            int s0 = ssegs[0];
            int n0 = snodes[0];
            Vec3 burg = network->segs[s0].burg;
            if (network->segs[s0].n2 == n0) burg = -burg;

            // Reject links with inconsistent glide planes
            Vec3 plane0 = network->segs[s0].plane;
            if (plane0.norm() < tol) continue;
            plane0 = plane0.normalized();

            bool consistent_plane = true;
            for (int k = 0; k < nseg; k++) {
                int sk = ssegs[k];
                Vec3 p = network->segs[sk].plane;
                if (p.norm() < tol) { consistent_plane = false; break; }
                p = p.normalized();
                if ((p - plane0).norm() > tol && (p + plane0).norm() > tol) {
                    consistent_plane = false; break;
                }
            }
            if (!consistent_plane) continue;

            // FCC filter: Burgers must be 1/2<110>, plane must be {111}
            Vec3 burg_crystal  = system->crystal.Rinv * burg.normalized();
            if (!is_110_family(burg_crystal)) continue;
            Vec3 plane_crystal = system->crystal.Rinv * plane0;
            if (!is_111_family(plane_crystal)) continue;

            // Screw character check
            int nfirst = snodes.front();
            int nlast  = snodes.back();
            Vec3 pfirst = network->nodes[nfirst].pos;
            Vec3 plast  = network->cell.pbc_position(pfirst,
                                                     network->nodes[nlast].pos);
            Vec3 chain_dir = plast - pfirst;
            double chain_len = chain_dir.norm();
            if (chain_len < tol) continue;
            chain_dir = chain_dir / chain_len;

            Vec3 bhat = burg.normalized();
            double screw_alignment = fabs(dot(chain_dir, bhat));
                        
            int non_screw_count = 0;
            for (int k = 0; k < nseg; k++) {
                int nk  = snodes[k];
                int nk1 = snodes[k+1];
                Vec3 pk  = network->nodes[nk].pos;
                Vec3 pk1 = network->cell.pbc_position(pk, network->nodes[nk1].pos);
                Vec3 seg_dir = pk1 - pk;
                double sl = seg_dir.norm();
                if (sl < tol) continue;
                if (fabs(dot(seg_dir / sl, bhat)) < scos) non_screw_count++;
            }
            if (screw_alignment < scos or non_screw_count > nseg / 2) continue;

            // Classify mechanism type
            MechanismType mechanism = Bulk;
            if (network->conn[nfirst].num > 2 ||
                network->conn[nlast].num  > 2) {
                mechanism = Intersection;
            }

            ScrewChain chain;
            chain.node_ids.assign(snodes.begin(), snodes.end());
            chain.seg_ids.assign(ssegs.begin(), ssegs.end());
            chain.burg        = burg;
            chain.glide_plane = plane0;
            chain.mechanism   = mechanism;

            // For Intersection: identify junction arm and junction type
            if (mechanism == Intersection) {
                int jnode = (network->conn[nfirst].num > 2) ? nfirst : nlast;

                Vec3 jburg(0.0);
                for (int k = 0; k < network->conn[jnode].num; k++) {
                    int sk = network->conn[jnode].seg[k];
                    bool in_chain = false;
                    for (int sc : ssegs) if (sc == sk) { in_chain = true; break; }
                    if (!in_chain) {
                        int ord = network->conn[jnode].order[k];
                        jburg = jburg + ord * network->segs[sk].burg;
                    }
                }
                chain.junction_burg = jburg;

                for (int k = 0; k < network->conn[jnode].num; k++) {
                    int sk = network->conn[jnode].seg[k];
                    bool in_chain = false;
                    for (int sc : ssegs) if (sc == sk) { in_chain = true; break; }
                    if (!in_chain) {
                        chain.junction_plane = network->segs[sk].plane;
                        break;
                    }
                }

                Vec3 jburg_crystal = system->crystal.Rinv * jburg.normalized();
                if (is_100_family(jburg_crystal)) {
                    chain.junction = Hirth;
                } else if (is_110_family(jburg_crystal)) {
                    Vec3 jplane_crystal = system->crystal.Rinv
                                         * chain.junction_plane.normalized();
                    if (is_111_family(jplane_crystal)) {
                        chain.junction = GlideLock;
                    } else if (is_110_family(jplane_crystal) or is_100_family(jplane_crystal)) {
                        chain.junction = LCLock;
                    }
                } else {
                    chain.junction = UnknownLock;
                }
            }

            chains.push_back(std::move(chain));
        }
        return chains;
    }

    // -----------------------------------------------------------------------
    //  Computes Schmid and Escaig stresses for a screw chain by averaging
    //  nodal forces over all segments.
    // -----------------------------------------------------------------------
    void compute_chain_stresses(SerialDisNet* network,
                                const ScrewChain& chain,
                                double burg_mag,
                                double& schmid_glide, double& schmid_cs,
                                double& escaig_glide, double& escaig_cs,
                                double& total_length)
    {
        Vec3 burg  = chain.burg;
        Vec3 plane = chain.glide_plane.normalized();
        // Ensure consistent sign convention for Schmid stress calculation
        if (dot(plane, cross(burg, plane)) < 0.0) plane = -plane;

        Vec3 cs_plane = get_crossslip_plane(plane, burg);
        if (cs_plane.norm() < 1e-6) {
            schmid_glide = schmid_cs = escaig_glide = escaig_cs = 0.0;
            total_length = 0.0;
            return;
        }
        if (dot(cs_plane, cross(burg, cs_plane)) < 0.0) cs_plane = -cs_plane;

        // Burgers direction for Escaig stress on each plane (Hussein et al. Sec. 2).
        Vec3 bhat = burg.normalized();
        Vec3 glide_bdir = bhat;
        Vec3 cs_bdir    = bhat;

        Vec3 total_force(0.0);
        total_length = 0.0;
        int nseg = (int)chain.seg_ids.size();
        for (int k = 0; k < nseg; k++) {
            int sk  = chain.seg_ids[k];
            int nk  = chain.node_ids[k];
            int nk1 = chain.node_ids[k+1];
            total_force  = total_force + 0.5 * (network->nodes[nk].f
                                               + network->nodes[nk1].f);
            total_length += network->seg_length(sk);
        }

        Vec3 glide_line = cross(burg, plane).normalized();
        Vec3 cs_line    = cross(burg, cs_plane).normalized();
        if (total_length > 1e-15) {
            schmid_glide = dot(total_force, glide_line) / total_length;
            schmid_cs    = dot(total_force, cs_line)    / total_length;
        } else {
            schmid_glide = schmid_cs = 0.0;
        }

        escaig_glide = compute_escaig_stress(total_force, glide_bdir, plane,    burg_mag, total_length);
        escaig_cs    = compute_escaig_stress(total_force, cs_bdir,    cs_plane, burg_mag, total_length);
    }

    // -----------------------------------------------------------------------
    //  Projects chain nodes onto the screw line and updates the glide plane.
    // -----------------------------------------------------------------------
    void execute_crossslip(SerialDisNet* network,
                           const ScrewChain& chain,
                           const Vec3& new_plane,
                           Mat33& dEp)
    {
        if (chain.node_ids.empty()) return;
        int nfirst = chain.node_ids.front();
        int nlast  = chain.node_ids.back();

        bool first_free = (network->nodes[nfirst].constraint == UNCONSTRAINED);
        bool last_free  = (network->nodes[nlast].constraint  == UNCONSTRAINED);

        Vec3 pivot(0.0);
        Vec3 ref_pos = network->nodes[nfirst].pos;
        if (first_free && last_free) {
            for (int nid : chain.node_ids){
                Vec3 p = network->cell.pbc_position(ref_pos, network->nodes[nid].pos);
                pivot = pivot + p;}
            pivot = pivot * (1.0 / (double)chain.node_ids.size());
        } else if (!first_free) {
            pivot = network->nodes[nfirst].pos;
        } else {
            pivot = network->nodes[nlast].pos;
        }

        Vec3 bhat = chain.burg.normalized();
        for (int i = 0; i < (int)chain.node_ids.size(); i++) {
            int  nid         = chain.node_ids[i];
            bool is_endpoint = (i == 0 || i == (int)chain.node_ids.size() - 1);
            if (is_endpoint) {
                bool moveable = (i == 0) ? first_free : last_free;
                if (!moveable) continue;
            }
            Vec3 pos  = network->cell.pbc_position(pivot, network->nodes[nid].pos);
            Vec3 proj = pivot + dot(pos - pivot, bhat) * bhat;
            network->move_node(nid, proj, dEp);
        }

        for (int sk : chain.seg_ids)
            update_seg_plane(network, sk, new_plane);
    }

    // -----------------------------------------------------------------------
    //  Bulk cross-slip: thermally activated, uses actual chain length.
    // -----------------------------------------------------------------------
    void handle_bulk(System* system, SerialDisNet* network,
                     const ScrewChain& chain)
    {
        double burg_mag = system->params.burgmag;
        double schmid_glide, schmid_cs, escaig_glide, escaig_cs, total_length;
        compute_chain_stresses(network, chain, burg_mag,
                               schmid_glide, schmid_cs,
                               escaig_glide, escaig_cs, total_length);
        if (total_length < 1e-15) return;

        // Stress threshold conditions (Hussein et al. Sec. 2):
        //   1. |tau_cs| >= mu*b / (10*L)
        //   2. |tau_cs| >= 1.1|tau_glide|  //   (Sheng: 1.1 is a tunable parameter to prevent cross-slip back and forth)
        double stress_threshold = system->params.MU * system->params.burgmag
                                  / (10.0 * total_length);
        if (fabs(schmid_cs) < stress_threshold)   return;
        if (fabs(schmid_cs) < 1.1*fabs(schmid_glide)) return;

        const Params& p = params;
        double V_a = p.bulkActivationVolume * burg_mag * burg_mag * burg_mag;
        double dSigma_E  = escaig_glide - escaig_cs;
        double dt        = p.evalFrequency * system->params.nextdt;
        double prob      = compute_thermal_probability(p.bulkActivationEnergy, V_a, dSigma_E,
                                                      p.temperature,
                                                      p.bulkAttemptFrequency, dt,
                                                      total_length,
                                                      p.bulkReferenceLength);
        if (prob < 1.0 && uniform01(rng) > prob) return;

        Vec3 plane = chain.glide_plane.normalized();
        if (dot(plane, cross(chain.burg, plane)) < 0.0) plane = -plane;
        Vec3 cs_plane = get_crossslip_plane(plane, chain.burg);
        if (cs_plane.norm() < 1e-6) return;
        execute_crossslip(network, chain, cs_plane, system->dEp);
    }

    // -----------------------------------------------------------------------
    //  Intersection cross-slip:
    //    Repulsive (spontaneous): executes immediately without thermal check.
    //    Attractive (thermally activated): uses fixed effective length and
    //    lock-type specific activation parameters.
    // -----------------------------------------------------------------------
    void handle_intersection(System* system, SerialDisNet* network,
                             const ScrewChain& chain)
    {
        const double tol = 1e-6;
        Vec3 plane = chain.glide_plane.normalized();
        if (dot(plane, cross(chain.burg, plane)) < 0.0) plane = -plane;
        Vec3 cs_plane = get_crossslip_plane(plane, chain.burg);
        if (cs_plane.norm() < tol) return;

        // Repulsive intersection (Hussein et al. Sec. 2): spontaneous and athermal
        // when junction Burgers vector sum is <112> type.
        Vec3 jburg_crystal = system->crystal.Rinv * chain.junction_burg.normalized();
        if (is_112_family(jburg_crystal)) {
            execute_crossslip(network, chain, cs_plane, system->dEp);
            return;
        }

        // Attractive: thermally activated, parameters depend on junction type
        double burg_mag = system->params.burgmag;
        const Params& p = params;
        double Ea, Va, gamma, freq, L_ref, eff_len;

        if (chain.junction == Hirth) {
            Ea      = p.hirthActivationEnergy;
            Va      = p.hirthActivationVolume * burg_mag * burg_mag * burg_mag;
            freq    = p.hirthAttemptFrequency;
            L_ref   = p.hirthReferenceLength;
            eff_len = p.hirthEffectiveLength / burg_mag;
        } else if (chain.junction == GlideLock) {
            Ea      = p.glideLockActivationEnergy;
            Va      = p.glideLockActivationVolume * burg_mag * burg_mag * burg_mag;
            freq    = p.glideLockAttemptFrequency;
            L_ref   = p.glideLockReferenceLength;
            eff_len = p.glideLockEffectiveLength / burg_mag;
        } else if (chain.junction == LCLock) {
            Ea      = p.lcLockActivationEnergy;
            Va      = p.lcLockActivationVolume * burg_mag * burg_mag * burg_mag;
            freq    = p.lcLockAttemptFrequency;
            L_ref   = p.lcLockReferenceLength;
            eff_len = p.lcLockEffectiveLength / burg_mag;
        } else {
            // Unknown junction type: fall back to bulk parameters
            Ea      = p.bulkActivationEnergy;
            Va      = p.bulkActivationVolume * burg_mag * burg_mag * burg_mag;
            freq    = p.bulkAttemptFrequency;
            L_ref   = p.bulkReferenceLength;
            eff_len = -1.0; // use actual chain length
        }

        double schmid_g, schmid_c, escaig_g, escaig_c, total_length;
        compute_chain_stresses(network, chain, burg_mag,
                               schmid_g, schmid_c, escaig_g, escaig_c,
                               total_length);

        double dSigma_E  = escaig_g - escaig_c;
        double dt        = p.evalFrequency * system->params.nextdt;
        double use_len   = (eff_len > 0.0) ? eff_len : total_length;
        double prob      = compute_thermal_probability(Ea, Va, dSigma_E,
                                                      p.temperature,
                                                      freq, dt, use_len, L_ref);
        if (prob < 1.0 && uniform01(rng) > prob) return;

        execute_crossslip(network, chain, cs_plane, system->dEp);
    }

public:
    CrossSlipFCCThermal() = default;
    CrossSlipFCCThermal(System* /*system*/, Force* _force, Params _params)
        : force(_force), params(_params),
          rng(std::random_device{}()) {}

    void handle(System* system) override
    {
        Kokkos::fence();
        system->timer[system->TIMER_CROSSSLIP].start();

        if (system->crystal.type != FCC_CRYSTAL)
            ExaDiS_fatal("Error: CrossSlipFCCThermal requires FCC_CRYSTAL\n");
        if (!system->crystal.use_glide_planes)
            ExaDiS_fatal("Error: CrossSlipFCCThermal requires use_glide_planes=true\n");

        eval_counter++;
        if (params.evalFrequency > 1 &&
            (eval_counter % params.evalFrequency) != 0) {
            system->timer[system->TIMER_CROSSSLIP].stop();
            return;
        }

        SerialDisNet* network = system->get_serial_network();

        std::vector<ScrewChain> chains = build_screw_chains(system, network);

        for (const ScrewChain& chain : chains) {
            switch (chain.mechanism) {
                case Bulk:         handle_bulk(system, network, chain);         break;
                case Intersection: handle_intersection(system, network, chain); break;
            }
        }

        Kokkos::fence();
        system->timer[system->TIMER_CROSSSLIP].stop();
    }

    const char* name() override { return "CrossSlipFCCThermal"; }
};


/*---------------------------------------------------------------------------
 *
 *    Class:    CrossSlipFCCThermalParallel<F>
 *              Kokkos-aware wrapper for CrossSlipFCCThermal.
 *              Adds Kokkos fencing around the serial algorithm for GPU builds.
 *
 *-------------------------------------------------------------------------*/
template<class F>
class CrossSlipFCCThermalParallel : public CrossSlipFCCThermal {
public:
    CrossSlipFCCThermalParallel(System* system, Force* _force,
                                CrossSlipFCCThermal::Params _params)
        : CrossSlipFCCThermal(system, _force, _params) {}

    void handle(System* system) override
    {
        Kokkos::fence();
        CrossSlipFCCThermal::handle(system);
        Kokkos::fence();
    }

    const char* name() override { return "CrossSlipFCCThermalParallel"; }
};

} // namespace ExaDiS

#endif // EXADIS_CROSS_SLIP_FCC_THERMAL_H