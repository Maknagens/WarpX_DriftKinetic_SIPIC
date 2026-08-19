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
}

#endif // WARPX_FLUX_TUBE_AREA_SCALING_H_
