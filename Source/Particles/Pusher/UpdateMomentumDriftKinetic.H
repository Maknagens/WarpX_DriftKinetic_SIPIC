/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_DRIFT_KINETIC_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_DRIFT_KINETIC_H_

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

/** \brief Advance a particle's axial momentum over one timestep using the
 *         drift-kinetic (guiding-center) equation of motion.
 *
 * The magnetic moment `mu = (ux^2 + uy^2) / (2 Bz)` is recomputed from the
 * current perpendicular velocity every step, rather than being stored. This
 * way any change to `(ux, uy)` made by the collision operator is picked up
 * automatically, without having to modify the collision operator.
 *
 * The axial momentum then receives the parallel electric force `q Ez / m` and
 * the magnetic-mirror force `-mu dBz/dz`:
 *
 *     uz += dt * ( q Ez / m - mu dBz/dz )
 *
 * With the fields gathered at the particle position, this is a time-centered
 * (2nd-order) leapfrog step. The perpendicular velocity is *not* touched here:
 * it is re-slaved to the invariant at the updated position by the caller,
 * after the position push (so that `mu` is conserved to 2nd order).
 *
 * \param ux, uy, uz  Particle momentum (proper velocity). Only uz is updated.
 * \param Ez          Axial electric field gathered to the particle.
 * \param Bz          Axial magnetic field gathered to the particle.
 * \param dBdz        Axial gradient dBz/dz gathered to the particle.
 * \param q           Charge of the particle (including ionization level).
 * \param m           Mass of the species.
 * \param dt          Time step size.
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumDriftKinetic (
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ez, const amrex::ParticleReal Bz,
    const amrex::ParticleReal dBdz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt )
{
    using namespace amrex::literals;

    // Magnetic moment (adiabatic invariant), recomputed from the current
    // perpendicular velocity so that collisions on (ux,uy) are respected.
    const amrex::ParticleReal v_perp_sq = ux*ux + uy*uy;
    const amrex::ParticleReal mu = (Bz > 0.0_prt) ? 0.5_prt * v_perp_sq / Bz : 0.0_prt;

    // Parallel push: electric force plus magnetic-mirror force.
    uz += dt * ( q * Ez / m - mu * dBdz );
}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_DRIFT_KINETIC_H_
