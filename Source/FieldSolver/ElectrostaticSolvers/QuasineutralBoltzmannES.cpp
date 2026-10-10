/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "QuasineutralBoltzmannES.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Fluids/MultiFluidContainer_fwd.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>

#include <cmath>

using namespace amrex;

void QuasineutralBoltzmannES::ReadBoltzmannParameters ()
{
    const ParmParse pp_warpx("warpx");

    utils::parser::queryWithParser(pp_warpx, "boltzmann_Te", m_Te);
    utils::parser::queryWithParser(pp_warpx, "boltzmann_n_ref", m_n_ref);
    utils::parser::queryWithParser(pp_warpx, "boltzmann_density_floor", m_density_floor);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_Te > 0._rt,
        "warpx.boltzmann_Te must be positive: it is the electron temperature "
        "in eV in n_e = n_ref exp(e phi / T_e).");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_n_ref > 0._rt,
        "warpx.boltzmann_n_ref must be positive (it is the argument of a "
        "logarithm). Its value only shifts phi by a constant.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_density_floor > 0._rt,
        "warpx.boltzmann_density_floor must be positive: phi = ln(n) is "
        "unbounded below as n -> 0, so a floor is required.");
}

void QuasineutralBoltzmannES::InitData ()
{
    auto & warpx = WarpX::GetInstance();
    m_poisson_boundary_handler->DefinePhiBCs(warpx.Geom(0));

    amrex::Print() << "QuasineutralBoltzmannES: T_e = " << m_Te << " eV"
                   << ", n_ref = " << m_n_ref << " m^-3"
                   << ", density_floor = " << m_density_floor << " m^-3\n";
    amrex::Print() << "QuasineutralBoltzmannES: no Poisson solve; phi is set by "
                      "quasineutrality, so potential boundary conditions and "
                      "the sheath are not represented.\n";
}

void QuasineutralBoltzmannES::ComputePhiFromDensity (
    amrex::MultiFab const & rho, amrex::MultiFab & phi, const int lev ) const
{
    // rho is the charge density summed over ion species, so n_i = rho/e gives
    // the electron density demanded by quasineutrality for any mix of charge
    // states. In the drift-kinetic flux tube the deposition has already
    // divided out A(z), so this is the physical 3D density.
    const Real inv_q_e   = 1._rt / PhysConst::q_e;
    const Real Te        = m_Te;          // [eV], so T_e/e is Te volts
    const Real n_ref     = m_n_ref;
    const Real n_floor   = m_density_floor;

    // Evaluate phi on the valid region only. The guard cells are filled
    // afterwards: ApplyFilterandSumBoundaryRho leaves rho correct in the guard
    // cells that overlap a neighbouring box, but the layer outside the
    // *physical* domain has no plasma in it, so evaluating the logarithm there
    // would clamp to the floor and hand computeE -- which differences
    // phi(i+1) - phi(i-1) at the boundary node -- an enormous spurious field.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box bx = mfi.tilebox();

        Array4<Real const> const & rho_arr = rho.const_array(mfi);
        Array4<Real>       const & phi_arr = phi.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            const Real n_i = amrex::max(n_floor, rho_arr(i,j,k) * inv_q_e);
            phi_arr(i,j,k) = Te * std::log(n_i / n_ref);
        });
    }

    // Box-to-box and periodic guard cells come from a neighbour's valid data.
    auto & warpx = WarpX::GetInstance();
    phi.FillBoundary(warpx.Geom(lev).periodicity());

    // At a physical boundary there is no neighbour, so extrapolate phi with
    // zero gradient. That gives E = 0 at the wall, which is the honest
    // statement for a quasineutral model: the sheath is below the resolution
    // of the approximation, and inventing a field from an empty guard cell
    // would be worse than admitting there is none.
    const amrex::Box domain = amrex::convert(warpx.Geom(lev).Domain(),
                                             phi.ixType().toIntVect());
    const int dom_lo = domain.smallEnd(0);
    const int dom_hi = domain.bigEnd(0);
    const bool periodic = warpx.Geom(lev).periodicity().isPeriodic(0);

    if (!periodic)
    {
        const amrex::IntVect ng = phi.nGrowVect();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box gbx = mfi.growntilebox(ng);
            Array4<Real> const & phi_arr = phi.array(mfi);

            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (i < dom_lo) { phi_arr(i,j,k) = phi_arr(dom_lo,j,k); }
                if (i > dom_hi) { phi_arr(i,j,k) = phi_arr(dom_hi,j,k); }
            });
        }
    }
}

void QuasineutralBoltzmannES::ComputeSpaceChargeField (
    ablastr::fields::MultiFabRegister& fields,
    MultiParticleContainer& mpc,
    MultiFluidContainer* mfl,
    int max_level)
{
    ABLASTR_PROFILE("QuasineutralBoltzmannES::ComputeSpaceChargeField");

    using ablastr::fields::MultiLevelScalarField;
    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mfl == nullptr,
        "QuasineutralBoltzmannES does not support fluid species.");

    const MultiLevelScalarField rho_fp = fields.get_mr_levels(FieldType::rho_fp, max_level);
    const MultiLevelScalarField phi_fp = fields.get_mr_levels(FieldType::phi_fp, max_level);
    const MultiLevelVectorField Efield_fp = fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);

    auto & warpx = WarpX::GetInstance();

    // Deposit the ion charge density. Going through the per-species
    // GetChargeDensity path applies the filter, the guard-cell sum, the
    // boundary reflection and -- in a flux tube -- the division by A(z), so
    // rho_fp comes out as the physical 3D charge density.
    for (const auto& rho_lev : rho_fp) { rho_lev->setVal(0.0_rt); }

    const int lev = 0;
    auto& mypc = warpx.GetPartContainer();
    for (auto const& pc : mypc) {
        if (pc->do_not_deposit) { continue; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pc->getCharge() > 0._rt,
            "QuasineutralBoltzmannES expects only positively charged (ion) "
            "species: the electrons are a Boltzmann fluid, not particles. "
            "Remove the electron species or set do_not_deposit for it.");

        auto rho = pc->GetChargeDensity(lev, true);
        warpx.ApplyFilterandSumBoundaryRho(lev, lev, *rho, 0, rho->nComp());

        auto add_ng = rho_fp[lev]->nGrowVect();
        add_ng.min(rho->nGrowVect());
        amrex::MultiFab::Add(*rho_fp[lev], *rho, 0, 0, 1, add_ng);
    }

    // phi = (T_e/e) ln(n_i/n_ref), evaluated pointwise. No solve.
    for (int ilev = 0; ilev < num_levels; ++ilev) {
        ComputePhiFromDensity(*rho_fp[ilev], *phi_fp[ilev], ilev);
    }

    // E = -grad(phi) by the usual centred difference, so the force on an ion
    // is the exact discrete gradient of the discrete potential.
    if (!EB::enabled()) {
        const std::array<Real, 3> beta = {0._rt};
        computeE( Efield_fp, phi_fp, beta );
    } else {
        WARPX_ABORT_WITH_MESSAGE(
            "QuasineutralBoltzmannES does not support embedded boundaries.");
    }

    amrex::ignore_unused(mpc);
}
