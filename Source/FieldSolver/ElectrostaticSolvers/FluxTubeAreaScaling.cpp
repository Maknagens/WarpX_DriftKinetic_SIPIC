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

#include <AMReX_Algorithm.H>
#include <AMReX_Box.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <utility>

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

        // Scale the valid region and every guard layer rho actually has. The
        // deposit can reach further than the external Bz has data -- the guard
        // widths come from different quantities in GuardCellManager and need
        // not line up (with algo.particle_shape = 2, for instance, rho gets
        // ng_alloc_J+1 = 3 layers while Bz gets ngz rounded up to even = 2) --
        // so A is read with a clamped index out there. B varies on the scale
        // of the mirror, thousands of cells, so taking the nearest available
        // node is a sub-percent error on the small charge in that outermost
        // layer, and far better than leaving it unscaled.
        const amrex::IntVect ng_scale = rho->nGrowVect();

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
            const amrex::Box gbx = mfi.growntilebox(ng_scale);

            amrex::Array4<amrex::Real> const & rho_arr = rho->array(mfi);
            amrex::Array4<const amrex::Real> const & Bz_arr = Bz.const_array(mfi);

            const amrex::Box bz_box = Bz[mfi].box();
            const int bz_lo = bz_box.smallEnd(0);
            const int bz_hi = bz_box.bigEnd(0);

            amrex::ParallelFor(gbx, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    const int ic = amrex::min(amrex::max(i, bz_lo), bz_hi);
                    const amrex::Real B = Bz_arr(ic,j,k);
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

amrex::IntVect ApplyFluxTubeFilter (
    amrex::MultiFab& dst, const amrex::MultiFab& src_mf,
    [[maybe_unused]] const int lev,
    const int scomp, const int dcomp, const int ncomp)
{
    using namespace amrex::literals;

    const amrex::IntVect ng_src = src_mf.nGrowVect();

#if defined(WARPX_DIM_1D_Z)
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    auto & warpx = WarpX::GetInstance();
    const int npass = static_cast<int>(WarpX::filter_npass_each_dir[0]);

    if (FluxTubeScalingEnabled() && npass > 0)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.m_fields.has(FieldType::Bfield_fp_external, Direction{2}, lev),
            "The flux-tube filter requires an external grid B field to define A(z).");

        const amrex::MultiFab & Bz =
            *warpx.m_fields.get(FieldType::Bfield_fp_external, Direction{2}, lev);
        const amrex::Real Bref = ReferenceB();

        // Each pass consumes one defined guard layer while pushing mass one
        // layer outward, so the working arrays are grown by 2*npass: after all
        // passes the defined region is still src.ng + npass, which covers the
        // reach of the filter. This runs before any guard-cell sum, so the
        // source guard layers hold local deposits only and are genuinely zero
        // beyond src.ng.
        const amrex::IntVect npass_vec(AMREX_D_DECL(npass, npass, npass));
        const amrex::IntVect ng_tmp = ng_src + 2*npass_vec;
        amrex::MultiFab tmp_a(src_mf.boxArray(), src_mf.DistributionMap(), ncomp, ng_tmp);
        amrex::MultiFab tmp_b(src_mf.boxArray(), src_mf.DistributionMap(), ncomp, ng_tmp);
        tmp_a.setVal(0.0_rt);
        tmp_b.setVal(0.0_rt);
        amrex::MultiFab::Copy(tmp_a, src_mf, scomp, 0, ncomp, ng_src);

        amrex::IntVect ng_avail = ng_tmp;

        // The node at bigEnd owns the outward face on the physical boundary;
        // zeroing the flux there keeps the volume integral over the valid
        // domain exactly conserved. A periodic direction keeps the ordinary
        // flux, since the guard sum restores it.
        const amrex::Box domain_t =
            amrex::convert(warpx.Geom(lev).Domain(), src_mf.ixType().toIntVect());
        const int dom_lo = domain_t.smallEnd(0);
        const int dom_hi = domain_t.bigEnd(0);
        const bool dir_periodic = warpx.Geom(lev).periodicity().isPeriodic(0);

        auto sweep = [&] (amrex::MultiFab& out, const amrex::MultiFab& in)
        {
            amrex::IntVect ng_out = ng_avail;
            ng_out[0] = std::max(0, ng_out[0] - 1);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(in, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                amrex::Box tb = amrex::convert(mfi.validbox(), in.ixType().toIntVect());
                tb.grow(ng_out);

                amrex::Array4<amrex::Real const> const& u = in.const_array(mfi);
                amrex::Array4<amrex::Real>       const& v = out.array(mfi);
                amrex::Array4<amrex::Real const> const& Bz_arr = Bz.const_array(mfi);

                // A is read with a clamped index where Bz has no data. The
                // filtered array is identically zero out there (deposits never
                // reach it), so the weight used cannot influence the result.
                const amrex::Box bz_box = Bz[mfi].box();
                const int bz_lo = bz_box.smallEnd(0);
                const int bz_hi = bz_box.bigEnd(0);

                auto point_weight = [=] AMREX_GPU_DEVICE (int i) -> amrex::Real
                {
                    const int ic = amrex::min(amrex::max(i, bz_lo), bz_hi);
                    const amrex::Real B = Bz_arr(ic,0,0);
                    return (B > 0._rt) ? Bref / B : 1._rt;
                };

                amrex::ParallelFor(tb, ncomp,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    const amrex::Real w0 = point_weight(i);
                    amrex::Real w_lo = 0.5_rt*(point_weight(i-1) + w0);
                    amrex::Real w_hi = 0.5_rt*(w0 + point_weight(i+1));
                    if (!dir_periodic) {
                        if (i >= dom_hi) { w_hi = 0._rt; }
                        if (i <= dom_lo) { w_lo = 0._rt; }
                        if (i >  dom_hi) { w_lo = 0._rt; }
                        if (i <  dom_lo) { w_hi = 0._rt; }
                    }
                    v(i,j,k,n) = u(i,j,k,n) + 0.25_rt/w0 *
                        ( w_hi*(u(i+1,j,k,n) - u(i,j,k,n))
                        - w_lo*(u(i,j,k,n) - u(i-1,j,k,n)) );
                });
            }
            ng_avail = ng_out;
        };

        amrex::MultiFab* in  = &tmp_a;
        amrex::MultiFab* out = &tmp_b;
        for (int p = 0; p < npass; ++p) {
            sweep(*out, *in);
            std::swap(in, out);
        }

        const amrex::IntVect ng_copy = amrex::min(dst.nGrowVect(), ng_avail);
        amrex::MultiFab::Copy(dst, *in, 0, dcomp, ncomp, ng_copy);
        return ng_copy;
    }
#endif

    // Not a flux-tube run (or no passes requested): pass the data through.
    const amrex::IntVect ng_copy = amrex::min(dst.nGrowVect(), ng_src);
    amrex::MultiFab::Copy(dst, src_mf, scomp, dcomp, ncomp, ng_copy);
    return ng_copy;
}

}
