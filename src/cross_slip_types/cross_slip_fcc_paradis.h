/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Thermally-activated cross-slip for FCC crystals.
 *  Reference: Hussein, Ahmed M., et al. Acta Materialia 85 (2015): 180-190.
 *
 *  The algorithm groups connected screw segments into "chains" and evaluates
 *  thermodynamic transition probabilities driven by Escaig stress differences
 *  between the glide and cross-slip planes. Three chain types are supported:
 *    - Bulk:         chains fully inside the simulation volume
 *    - Surface:      chains with at least one surface-pinned endpoint
 *    - Intersection: chains terminating at dislocation junctions (Hirth,
 *                    Glide lock, LC lock)
 *
 *  Thermal probability:
 *    P = nu * dt * (L / L_ref) * exp(-(E_a - V_a * dSigma_Escaig) / (kB * T))
 *  where E_a and V_a are SFE-scaled activation energy and volume.
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
        double bulkActivationEnergy       = 2.77e-19; ///< E_a [J]
        double bulkActivationVolumeFactor = 9.0;      ///< V_a = factor * b^3
        double bulkStackingFaultEnergy    = 0.04;     ///< gamma [J/m^2]
        double bulkDissociationRatio      = 8.0;      ///< d/b, partial separation ratio
        double bulkAttemptFrequency       = 1e11;     ///< nu [1/s]
        double bulkReferenceLength        = 1000.0;   ///< L_ref [b]

        // --- Surface cross-slip ---
        double surfaceActivationEnergy       = 2.77e-19;
        double surfaceActivationVolumeFactor = 9.0;
        double surfaceStackingFaultEnergy    = 0.04;
        double surfaceDissociationRatio      = 8.0;
        double surfaceAttemptFrequency       = 1e11;
        double surfaceReferenceLength        = 1000.0;
        double surfaceEffectiveLength        = 100.0; ///< Fixed chain length for surface [b]

        // --- Hirth lock (intersection) ---
        double hirthActivationEnergy       = 2.77e-19;
        double hirthActivationVolumeFactor = 9.0;
        double hirthStackingFaultEnergy    = 0.04;
        double hirthDissociationRatio      = 8.0;
        double hirthAttemptFrequency       = 1e11;
        double hirthReferenceLength        = 1000.0;
        double hirthEffectiveLength        = 100.0;

        // --- Glide lock (intersection) ---
        double glideLockActivationEnergy       = 2.77e-19;
        double glideLockActivationVolumeFactor = 9.0;
        double glideLockStackingFaultEnergy    = 0.04;
        double glideLockDissociationRatio      = 8.0;
        double glideLockAttemptFrequency       = 1e11;
        double glideLockReferenceLength        = 1000.0;
        double glideLockEffectiveLength        = 100.0;

        // --- LC lock / Lomer-Cottrell lock (intersection) ---
        double lcLockActivationEnergy       = 2.77e-19;
        double lcLockActivationVolumeFactor = 9.0;
        double lcLockStackingFaultEnergy    = 0.04;
        double lcLockDissociationRatio      = 8.0;
        double lcLockAttemptFrequency       = 1e11;
        double lcLockReferenceLength        = 1000.0;
        double lcLockEffectiveLength        = 100.0;

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

    enum MechanismType { Bulk, Surface, Intersection };
    enum JunctionType  { Hirth, GlideLock, LCLock, UnknownJunction };

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
        JunctionType  junction  = UnknownJunction;
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
                                        double      burg_mag) {
        if (burg_mag < 1e-15) return 0.0;
        return dot(plane_normal, force) * dot(plane_normal, burg_dir) / burg_mag;
    }

    // -----------------------------------------------------------------------
    //  SFE-scaled activation energy:
    //  E_a_scaled = E_a * (gamma / gamma_eff) * sqrt(log(d/b_eff) / log(d/b_ref))
    // -----------------------------------------------------------------------
    static double scale_activation_energy(double E_a,
                                          double dissociation_ratio,
                                          double gamma_ref,
                                          double gamma_eff) {
        const double tol = 1e-6;
        if (gamma_eff < tol) gamma_eff = tol;
        double d_b_eff = dissociation_ratio * gamma_ref / gamma_eff;
        if (d_b_eff < 1.0) d_b_eff = 1.0;
        if (dissociation_ratio <= 1.0 || d_b_eff <= 1.0) return E_a;
        return E_a * (gamma_ref / gamma_eff)
               * sqrt(log(d_b_eff) / log(dissociation_ratio));
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
            Vec3 burg0 = network->segs[s0].burg;
            if (network->segs[s0].n2 == n0) burg0 = -burg0;

            // Reject links with inconsistent Burgers vectors
            bool consistent_burg = true;
            for (int k = 0; k < nseg; k++) {
                int sk = ssegs[k];
                int nk = snodes[k];
                Vec3 b = network->segs[sk].burg;
                if (network->segs[sk].n2 == nk) b = -b;
                if ((b - burg0).norm() > tol && (b + burg0).norm() > tol) {
                    consistent_burg = false; break;
                }
            }
            if (!consistent_burg) continue;

            Vec3 burg = burg0;

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

            // Allow up to 3 non-screw segments in an otherwise screw chain
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
            if (non_screw_count > 3 && screw_alignment < scos) continue;
            if (screw_alignment < scos && non_screw_count > nseg / 2) continue;

            // Classify mechanism type
            MechanismType mechanism = Bulk;
            if (network->nodes[nfirst].constraint == PINNED_NODE ||
                network->nodes[nfirst].constraint == CORNER_NODE ||
                network->nodes[nlast].constraint  == PINNED_NODE ||
                network->nodes[nlast].constraint  == CORNER_NODE) {
                mechanism = Surface;
            }
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
                    chain.junction = is_111_family(jplane_crystal) ? GlideLock : LCLock;
                } else {
                    chain.junction = UnknownJunction;
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
        if (plane.x * plane.y * plane.z < 0.0) plane = -plane;

        Vec3 cs_plane = get_crossslip_plane(plane, burg);
        if (cs_plane.norm() < 1e-6) {
            schmid_glide = schmid_cs = escaig_glide = escaig_cs = 0.0;
            total_length = 0.0;
            return;
        }
        if (cs_plane.x * cs_plane.y * cs_plane.z < 0.0) cs_plane = -cs_plane;

        // Sign-correct Burgers direction for Escaig stress computation
        Vec3 bhat = burg.normalized();
        const double tol = 1e-6;

        Vec3 glide_bdir = bhat;
        Vec3 glide_temp = cross(plane, bhat);
        bool rev = false;
        if (fabs(glide_bdir.x) < tol && glide_temp.x * plane.x < 0.0) rev = true;
        if (fabs(glide_bdir.y) < tol && glide_temp.y * plane.y < 0.0) rev = true;
        if (fabs(glide_bdir.z) < tol && glide_temp.z * plane.z < 0.0) rev = true;
        if (rev) glide_bdir = -glide_bdir;

        Vec3 cs_bdir = bhat;
        Vec3 cs_temp = cross(cs_plane, bhat);
        rev = false;
        if (fabs(cs_bdir.x) < tol && cs_temp.x * cs_plane.x < 0.0) rev = true;
        if (fabs(cs_bdir.y) < tol && cs_temp.y * cs_plane.y < 0.0) rev = true;
        if (fabs(cs_bdir.z) < tol && cs_temp.z * cs_plane.z < 0.0) rev = true;
        if (rev) cs_bdir = -cs_bdir;

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

        escaig_glide = compute_escaig_stress(total_force, glide_bdir, plane,    burg_mag);
        escaig_cs    = compute_escaig_stress(total_force, cs_bdir,    cs_plane, burg_mag);
    }

    // -----------------------------------------------------------------------
    //  Projects chain nodes onto the screw line and updates the glide plane.
    //  Pinned/corner nodes are not moved.
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
        if (first_free && last_free) {
            for (int nid : chain.node_ids)
                pivot = pivot + network->nodes[nid].pos;
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
            Vec3 pos  = network->nodes[nid].pos;
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

        double stress_threshold = system->params.MU / (2.0 * total_length);
        if (fabs(schmid_cs) < stress_threshold)          return;
        if (fabs(schmid_cs) < 0.5 * fabs(schmid_glide)) return;

        const Params& p = params;
        double V_a       = p.bulkActivationVolumeFactor * burg_mag * burg_mag * burg_mag;
        double b_edge    = burg_mag / (2.0 * sqrt(3.0));
        double gamma_eff = p.bulkStackingFaultEnergy + escaig_cs * b_edge;
        double E_a_eff   = scale_activation_energy(p.bulkActivationEnergy,
                                                   p.bulkDissociationRatio,
                                                   p.bulkStackingFaultEnergy, gamma_eff);
        double dSigma_E  = escaig_glide - escaig_cs;
        double dt        = p.evalFrequency * system->params.nextdt;
        double prob      = compute_thermal_probability(E_a_eff, V_a, dSigma_E,
                                                      p.temperature,
                                                      p.bulkAttemptFrequency, dt,
                                                      total_length,
                                                      p.bulkReferenceLength);
        if (prob < 1.0 && uniform01(rng) > prob) return;

        Vec3 plane = chain.glide_plane.normalized();
        if (plane.x * plane.y * plane.z < 0.0) plane = -plane;
        Vec3 cs_plane = get_crossslip_plane(plane, chain.burg);
        if (cs_plane.norm() < 1e-6) return;
        execute_crossslip(network, chain, cs_plane, system->dEp);
    }

    // -----------------------------------------------------------------------
    //  Surface cross-slip: uses fixed effective chain length (surfaceEffectiveLength).
    // -----------------------------------------------------------------------
    void handle_surface(System* system, SerialDisNet* network,
                        const ScrewChain& chain)
    {
        double burg_mag = system->params.burgmag;
        double schmid_glide, schmid_cs, escaig_glide, escaig_cs, total_length;
        compute_chain_stresses(network, chain, burg_mag,
                               schmid_glide, schmid_cs,
                               escaig_glide, escaig_cs, total_length);
        if (total_length < 1e-15) return;

        const Params& p = params;
        double stress_threshold = system->params.MU / (4.0 * total_length);
        if (fabs(schmid_cs) < stress_threshold)          return;
        if (fabs(schmid_cs) < 0.5 * fabs(schmid_glide)) return;

        double V_a       = p.surfaceActivationVolumeFactor * burg_mag * burg_mag * burg_mag;
        double b_edge    = burg_mag / (2.0 * sqrt(3.0));
        double gamma_eff = p.surfaceStackingFaultEnergy + escaig_cs * b_edge;
        double E_a_eff   = scale_activation_energy(p.surfaceActivationEnergy,
                                                   p.surfaceDissociationRatio,
                                                   p.surfaceStackingFaultEnergy, gamma_eff);
        double dSigma_E  = escaig_glide - escaig_cs;
        double dt        = p.evalFrequency * system->params.nextdt;
        double prob      = compute_thermal_probability(E_a_eff, V_a, dSigma_E,
                                                      p.temperature,
                                                      p.surfaceAttemptFrequency, dt,
                                                      p.surfaceEffectiveLength,
                                                      p.surfaceReferenceLength);
        if (prob < 1.0 && uniform01(rng) > prob) return;

        Vec3 plane = chain.glide_plane.normalized();
        if (plane.x * plane.y * plane.z < 0.0) plane = -plane;
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
        if (plane.x * plane.y * plane.z < 0.0) plane = -plane;
        Vec3 cs_plane = get_crossslip_plane(plane, chain.burg);
        if (cs_plane.norm() < tol) return;

        Vec3 jplane = chain.junction_plane;
        if (jplane.norm() < tol) return;
        jplane = jplane.normalized();

        Vec3 glide_junc_line = cross(plane,    jplane);
        Vec3 cs_junc_line    = cross(cs_plane, jplane);
        if (glide_junc_line.norm() < tol || cs_junc_line.norm() < tol) return;

        int nfirst = chain.node_ids.front();
        int nlast  = chain.node_ids.back();
        int jnode  = (network->conn[nfirst].num > 2) ? nfirst : nlast;

        Vec3 jarm_dir(0.0);
        for (int k = 0; k < network->conn[jnode].num; k++) {
            int sk = network->conn[jnode].seg[k];
            bool in_chain = false;
            for (int sc : chain.seg_ids) if (sc == sk) { in_chain = true; break; }
            if (!in_chain) {
                int  nn  = network->conn[jnode].node[k];
                Vec3 pj  = network->nodes[jnode].pos;
                Vec3 pn  = network->cell.pbc_position(pj, network->nodes[nn].pos);
                Vec3 d   = pn - pj;
                if (d.norm() > tol) { jarm_dir = d.normalized(); break; }
            }
        }
        if (jarm_dir.norm() < tol) return;

        // Repulsive: spontaneous cross-slip (no thermal check)
        if (fabs(dot(glide_junc_line, jarm_dir)) < fabs(dot(cs_junc_line, jarm_dir))) {
            execute_crossslip(network, chain, cs_plane, system->dEp);
            return;
        }

        // Attractive: thermally activated, parameters depend on junction type
        double burg_mag = system->params.burgmag;
        const Params& p = params;
        double Ea, Va, gamma, d_b, freq, L_ref, eff_len;

        if (chain.junction == Hirth) {
            Ea      = p.hirthActivationEnergy;
            Va      = p.hirthActivationVolumeFactor * burg_mag * burg_mag * burg_mag;
            gamma   = p.hirthStackingFaultEnergy;
            d_b     = p.hirthDissociationRatio;
            freq    = p.hirthAttemptFrequency;
            L_ref   = p.hirthReferenceLength;
            eff_len = p.hirthEffectiveLength;
        } else if (chain.junction == GlideLock) {
            Ea      = p.glideLockActivationEnergy;
            Va      = p.glideLockActivationVolumeFactor * burg_mag * burg_mag * burg_mag;
            gamma   = p.glideLockStackingFaultEnergy;
            d_b     = p.glideLockDissociationRatio;
            freq    = p.glideLockAttemptFrequency;
            L_ref   = p.glideLockReferenceLength;
            eff_len = p.glideLockEffectiveLength;
        } else if (chain.junction == LCLock) {
            Ea      = p.lcLockActivationEnergy;
            Va      = p.lcLockActivationVolumeFactor * burg_mag * burg_mag * burg_mag;
            gamma   = p.lcLockStackingFaultEnergy;
            d_b     = p.lcLockDissociationRatio;
            freq    = p.lcLockAttemptFrequency;
            L_ref   = p.lcLockReferenceLength;
            eff_len = p.lcLockEffectiveLength;
        } else {
            // Unknown junction type: fall back to bulk parameters
            Ea      = p.bulkActivationEnergy;
            Va      = p.bulkActivationVolumeFactor * burg_mag * burg_mag * burg_mag;
            gamma   = p.bulkStackingFaultEnergy;
            d_b     = p.bulkDissociationRatio;
            freq    = p.bulkAttemptFrequency;
            L_ref   = p.bulkReferenceLength;
            eff_len = -1.0; // use actual chain length
        }

        double schmid_g, schmid_c, escaig_g, escaig_c, total_length;
        compute_chain_stresses(network, chain, burg_mag,
                               schmid_g, schmid_c, escaig_g, escaig_c,
                               total_length);

        double b_edge    = burg_mag / (2.0 * sqrt(3.0));
        double gamma_eff = gamma + escaig_c * b_edge;
        double E_a_eff   = scale_activation_energy(Ea, d_b, gamma, gamma_eff);
        double dSigma_E  = escaig_g - escaig_c;
        double dt        = p.evalFrequency * system->params.nextdt;
        double use_len   = (eff_len > 0.0) ? eff_len : total_length;
        double prob      = compute_thermal_probability(E_a_eff, Va, dSigma_E,
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
                case Surface:      handle_surface(system, network, chain);      break;
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