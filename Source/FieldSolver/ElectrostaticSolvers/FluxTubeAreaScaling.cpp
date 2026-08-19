/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FluxTubeAreaScaling.H"

#include "Fields.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpX.H"

#include <AMReX_Box.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>

namespace warpx::drift_kinetic
{

bool FluxTubeScalingEnabled ()
{
#if defined(WARPX_DIM_1D_Z)
    return (WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameDriftKinetic);
#else
    return false;
#endif
}

amrex::Real ReferenceB ()
{
    // The reference field is a static input but this is called from the
    // deposition path on every step, so parse it only once.
    static const amrex::Real B_ref = [] () {
        amrex::Real B = amrex::Real(1.0);
        const amrex::ParmParse pp_warpx("warpx");
        utils::parser::queryWithParser(pp_warpx, "drift_kinetic_reference_B", B);
        return B;
    }();
    return B_ref;
}

namespace
{
    /** \brief Scale a nodal charge density by a power of the flux-tube area.
     *
     * \param[in,out] rho      nodal charge density to rescale in place
     * \param[in]     lev      refinement level that `rho` lives on
     * \param[in]     multiply true multiplies by A(z), false divides by it
     */
    void ScaleByFluxTubeArea ([[maybe_unused]] amrex::MultiFab* rho,
                              [[maybe_unused]] const int lev,
                              [[maybe_unused]] const bool multiply)
    {
#if defined(WARPX_DIM_1D_Z)
        if (!FluxTubeScalingEnabled()) { return; }

        using ablastr::fields::Direction;
        using warpx::fields::FieldType;

        auto & warpx = WarpX::GetInstance();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.m_fields.has(FieldType::Bfield_fp_external, Direction{2}, lev),
            "The drift-kinetic flux-tube scaling requires an external grid B field "
            "(set it via picmi.AnalyticInitialField / B_ext_grid_init_style).");

        const amrex::MultiFab & Bz =
            *warpx.m_fields.get(FieldType::Bfield_fp_external, Direction{2}, lev);
        const amrex::Real Bref = ReferenceB();
        const int ncomp = rho->nComp();

        // Guard cells are scaled too (see below), so the area must be defined
        // everywhere charge was deposited. Checked once, outside the loop.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            Bz.nGrowVect().allGE(rho->nGrowVect()),
            "The external Bz field has fewer guard cells than the charge density, "
            "so the flux-tube area is undefined where charge was deposited. Lower "
            "the deposition shape order (algo.particle_shape) or allocate more "
            "field guard cells.");

        // The deposition kernels divide by the Cartesian cell volume (dz in 1D),
        // so they produce rho_1D = A rho_3D with A(z) = B_ref/B(z). Dividing by
        // A is therefore multiplying by B(z)/B_ref. Both rho and the external Bz
        // are nodal in z, so this is a pointwise scaling with no interpolation.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*rho, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Guard cells are scaled too: this runs before the guard-cell sum,
            // and A is a function of position, so a guard cell and the valid
            // cell it folds into see the same A.
            const amrex::Box gbx = mfi.growntilebox(rho->nGrowVect());

            amrex::Array4<amrex::Real> const & rho_arr = rho->array(mfi);
            amrex::Array4<const amrex::Real> const & Bz_arr = Bz.const_array(mfi);

            amrex::ParallelFor(gbx, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    const amrex::Real B = Bz_arr(i,j,k);
                    if (B > amrex::Real(0.)) {
                        rho_arr(i,j,k,n) *= multiply ? (Bref / B) : (B / Bref);
                    }
                });
        }
#endif
    }
}

void ScaleChargeDensityToFluxTubeVolume (amrex::MultiFab* rho, const int lev)
{
    ScaleByFluxTubeArea(rho, lev, /*multiply=*/false);
}

void ScaleChargeDensityByFluxTubeArea (amrex::MultiFab* rho, const int lev)
{
    ScaleByFluxTubeArea(rho, lev, /*multiply=*/true);
}

}
