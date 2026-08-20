/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FLUX_TUBE_AREA_SCALING_H_
#define WARPX_FLUX_TUBE_AREA_SCALING_H_

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

/** \brief Flux-tube volume scaling for the drift-kinetic mirror model.
 *
 * The 1D drift-kinetic model follows a magnetic flux tube whose cross-section
 * varies along the field line as A(z) = B_ref / B(z). The charge deposition
 * kernels are geometry-agnostic: they divide the particle weights by the
 * Cartesian cell volume (dz in 1D), so what they produce is the charge per
 * unit length along the field line,
 *
 *     rho_1D(z) = A(z) rho_3D(z).
 *
 * Every density-like consumer (the semi-implicit plasma frequency, the Debye
 * length, collision rates, density diagnostics) wants rho_3D, so the scaling
 * is applied once at the end of the deposition -- the same place and for the
 * same reason that radial geometries call
 * WarpX::ApplyInverseVolumeScalingToChargeDensity.
 *
 * The Poisson right-hand side is the exception: the flux-tube Poisson equation
 * d_z(A eps0 d_z phi) = -A rho_3D carries a factor A on the source term, so
 * the electrostatic solver multiplies rho back by A immediately before the
 * MLMG solve.
 */
namespace warpx::drift_kinetic
{
    /** \brief Whether the flux-tube volume scaling applies to this run.
     *
     * True only for a 1D drift-kinetic electrostatic run that has an external
     * grid Bz to define A(z) from.
     */
    bool FluxTubeScalingEnabled ();

    /** \brief The reference field B_ref in A(z) = B_ref/B(z), i.e. the field
     *         at the plane where the flux tube has unit cross-section.
     *
     * Read from `warpx.drift_kinetic_reference_B` (the same input parameter
     * that LabFrameDriftKineticES reads).
     */
    amrex::Real ReferenceB ();

    /** \brief Divide a deposited charge density by the flux-tube area A(z),
     *         converting charge-per-unit-length into a 3D charge density.
     *
     * A no-op unless FluxTubeScalingEnabled(). Operates on the valid region
     * and all guard cells, so it is safe to call before the guard-cell sum.
     *
     * \param[in,out] rho nodal charge density to rescale in place
     * \param[in]     lev refinement level that `rho` lives on
     */
    void ScaleChargeDensityToFluxTubeVolume (amrex::MultiFab* rho, int lev);

    /** \brief Multiply a 3D charge density by the flux-tube area A(z), the
     *         inverse of ScaleChargeDensityToFluxTubeVolume.
     *
     * Used to build the flux-tube Poisson source term A rho_3D from rho_3D.
     *
     * \param[in,out] rho nodal charge density to rescale in place
     * \param[in]     lev refinement level that `rho` lives on
     */
    void ScaleChargeDensityByFluxTubeArea (amrex::MultiFab* rho, int lev);

    /** \brief Volume-weighted binomial filter for the flux tube.
     *
     * The plain (1/4, 1/2, 1/4) stencil conserves the sum of the array, which
     * is the total charge only when every cell has the same volume. In the
     * flux tube the volume is A(z) dz, so filtering the 3D density moves
     * density from narrow cells into wide ones and changes the total charge.
     *
     * This is the flux form of the same pass,
     *
     *     v_i = u_i + (1/4)/w_i [ w_{i+1/2}(u_{i+1} - u_i)
     *                           - w_{i-1/2}(u_i - u_{i-1}) ],
     *
     * with w the cell volume factor A(z) and w_{i+-1/2} the face averages.
     * Written this way the update is the divergence of a two-point diffusive
     * flux, so sum_i w_i u_i -- the total charge -- telescopes and is
     * conserved exactly, for any A(z). It reduces to (1/4, 1/2, 1/4) where A
     * is uniform, leaves constants untouched, and carries no flux through the
     * physical domain boundaries.
     *
     * Mirrors WarpX::ApplyVolumeWeightedFilter, which does the same thing for
     * radial geometries, and applies `warpx.filter_npass_each_dir` passes.
     *
     * \param[out] dst     filtered result
     * \param[in]  src_mf  unfiltered source
     * \param[in]  lev     refinement level
     * \param[in]  scomp   first source component
     * \param[in]  dcomp   first destination component
     * \param[in]  ncomp   number of components
     * \return the number of guard layers of `dst` that hold filtered data
     */
    amrex::IntVect ApplyFluxTubeFilter (
        amrex::MultiFab& dst, const amrex::MultiFab& src_mf,
        int lev, int scomp, int dcomp, int ncomp);
}

#endif // WARPX_FLUX_TUBE_AREA_SCALING_H_
