/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_LABFRAMEDRIFTKINETICES_H_
#define WARPX_LABFRAMEDRIFTKINETICES_H_

#include "ElectrostaticSolver.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/fields/EffectivePotentialPoissonSolver.H>

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>

/**
 * \brief Semi-implicit (effective-potential) electrostatic solver for the 1D
 *        drift-kinetic magnetic-mirror model.
 *
 * Solves the variable-coefficient Poisson equation
 * \f[
 *     \nabla\cdot\left(\sigma\,\nabla\phi\right) = -\frac{\rho}{\varepsilon_0}
 * \f]
 * with \f$\sigma = A(z)\,\varepsilon_{SI}\f$, where \f$A(z) = B_{ref}/B(z)\f$ is
 * the (time-constant) magnetic flux-tube cross-section and \f$\varepsilon_{SI}\f$
 * is the effective-potential semi-implicit dressing (Barnes, J. Comp. Phys. 424,
 * 2021), identical to EffectivePotentialES. Setting the semi-implicit factor to
 * zero recovers the explicit drift-kinetic Poisson equation.
 *
 * The right (z_hi) wall type follows the z_hi field boundary condition:
 *  - Neumann   (`boundary.field_hi = neumann`): a floating wall. Charge
 *    collected there (read from the ParticleBoundaryBuffer) is accumulated and
 *    injected into rho at the boundary node -- a charge-sheet (dynamic Neumann)
 *    floating wall.
 *  - Dirichlet (`boundary.field_hi = pec`): a fixed-potential wall held at
 *    `boundary.potential_hi_z` (default 0); no charge sheet is injected.
 * Either way the z_hi particle buffer is drained every step. The left (z_lo)
 * wall is grounded (Dirichlet phi = 0).
 */
class LabFrameDriftKineticES final : public ElectrostaticSolver
{
public:

    LabFrameDriftKineticES (int nlevs_max) : ElectrostaticSolver (nlevs_max) {
        ReadParameters();
        ReadDriftKineticParameters();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (nlevs_max == 1),
            "LabFrameDriftKineticES only supports one level at present"
        );
    }

    void InitData () override;

    void ComputeSpaceChargeField (
        ablastr::fields::MultiFabRegister& fields,
        MultiParticleContainer& mpc,
        MultiFluidContainer* mfl,
        int max_level) override;

    /** \brief Build sigma = A(z) * eps_SI (flux-tube area times the
     *         semi-implicit effective-potential dressing). */
    void ComputeSigma ( amrex::MultiFab& sigma ) const;

    void computePhi (
        ablastr::fields::MultiLevelScalarField const& rho,
        ablastr::fields::MultiLevelScalarField const& phi,
        ablastr::fields::MultiLevelVectorField const& efield
    ) const;

    void computePhi (
        ablastr::fields::MultiLevelScalarField const& rho,
        ablastr::fields::MultiLevelScalarField const& phi,
        ablastr::fields::MultiLevelVectorField const& efield,
        amrex::MultiFab const& sigma,
        amrex::Real required_precision,
        amrex::Real absolute_tolerance,
        int max_iters,
        int verbosity
    ) const;

private:

    /** \brief Read drift-kinetic / floating-wall input parameters. */
    void ReadDriftKineticParameters ();

    /** \brief Accumulate the charge collected at the floating z_hi wall into
     *         m_sigma_wall and return it. Clears the z_hi boundary buffer. */
    amrex::Real AccumulateWallCharge ();

    /** \brief Append (I/O rank only) the current step and m_sigma_wall to the
     *         floating-wall state log. Called once per step. */
    void LogWallCharge () const;

    /** \brief Seed m_sigma_wall on restart from the state log (matching the
     *         checkpoint step parsed from `amr.restart`); on a fresh run,
     *         truncate the log. Called from InitData. This keeps the floating
     *         wall charge checkpoint-compatible without touching core WarpX
     *         checkpoint I/O. */
    void RestoreWallCharge ();

    /** Reference magnetic field B_ref (A = 1 where B = B_ref). */
    amrex::Real m_reference_B = 1.0;

    /** Semi-implicit (effective-potential) factor C_SI; 0 => fully explicit. */
    amrex::Real m_effective_potential_factor = 0.0;

    /** Accumulated surface charge collected at the floating z_hi wall. */
    amrex::Real m_sigma_wall = 0.0;

    /** z_hi wall mode, set in InitData from the z_hi field BC: true => floating
     *  (Neumann + charge sheet), false => Dirichlet (fixed potential). */
    bool m_zhi_floating = true;

    /** Path of the floating-wall charge state log (checkpoint persistence). */
    std::string m_wall_charge_log = "drift_kinetic_wall_charge.txt";
};

#endif // WARPX_LABFRAMEDRIFTKINETICES_H_
