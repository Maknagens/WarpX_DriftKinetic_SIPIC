/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "LabFrameDriftKineticES.H"
#include "Fluids/MultiFluidContainer_fwd.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "FluxTubeAreaScaling.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

using namespace amrex;

void LabFrameDriftKineticES::ReadDriftKineticParameters ()
{
    const ParmParse pp_warpx("warpx");
    // Reference field B_ref: the flux-tube area is A(z) = B_ref / B(z), so
    // A = 1 (standard Poisson) where B = B_ref. Typically B at z_lo.
    utils::parser::queryWithParser(pp_warpx, "drift_kinetic_reference_B", m_reference_B);
    // Semi-implicit effective-potential factor; 0 recovers the explicit solve.
    utils::parser::queryWithParser(pp_warpx, "effective_potential_factor", m_effective_potential_factor);
    // Temporal filtering of the dressing (1 = instantaneous) and the density
    // floor below which the dressing is clamped. Same input names and defaults
    // as the upstream EffectivePotentialES solver.
    utils::parser::queryWithParser(pp_warpx, "effective_potential_time_filter_param", m_time_filter_param);
    utils::parser::queryWithParser(pp_warpx, "effective_potential_density_floor", m_density_floor);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_time_filter_param > 0._rt && m_time_filter_param <= 1._rt,
        "warpx.effective_potential_time_filter_param must be in (0, 1]: it is the "
        "weight of the new value in eps^n = (1-a) eps^{n-1} + a eps_new, so a = 1 "
        "means no filtering and a -> 0 freezes the dressing at its initial value.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_density_floor >= 0._rt,
        "warpx.effective_potential_density_floor must be non-negative.");
    // Optional override of the floating-wall charge state-log filename.
    pp_warpx.query("drift_kinetic_wall_charge_log", m_wall_charge_log);
}

void LabFrameDriftKineticES::InitData ()
{
    auto & warpx = WarpX::GetInstance();
    m_poisson_boundary_handler->DefinePhiBCs(warpx.Geom(0));

    // Echo the drift-kinetic parameters so it is visible whether the inputs
    // (warpx.effective_potential_factor, warpx.drift_kinetic_reference_B)
    // actually reached the solver.
    amrex::Print() << "LabFrameDriftKineticES: effective_potential_factor (C_SI) = "
                   << m_effective_potential_factor
                   << ", reference_B = " << m_reference_B
                   << ", time_filter_param = " << m_time_filter_param
                   << ", density_floor = " << m_density_floor << "\n";

    // Persistent store for the semi-implicit dressing eps_SI, i.e. sigma
    // without the flux-tube area. It lives in the MultiFab registry rather
    // than in a member so that it is remade when load balancing changes the
    // distribution mapping. A(z) is deliberately kept out of it: A is static
    // in time, so filtering it would be meaningless, and storing it would
    // reintroduce exactly the staleness the registry avoids.
    {
        auto & mf_registry = warpx.GetMultiFabRegister();
        if (!mf_registry.has(warpx::fields::FieldType::effective_potential_sigma, 0))
        {
            const amrex::MultiFab * rho_ptr =
                mf_registry.get(warpx::fields::FieldType::rho_fp, 0);
            mf_registry.alloc_init(
                warpx::fields::FieldType::effective_potential_sigma, /*level=*/ 0,
                amrex::convert(rho_ptr->boxArray(), amrex::IntVect(AMREX_D_DECL(0,0,0))),
                rho_ptr->DistributionMap(), 1, amrex::IntVect(AMREX_D_DECL(0,0,0)), 1.0_rt
            );
        }
    }
    // There is no history on the first step, nor after a restart (InitData
    // runs in both cases), so seed the filter from the instantaneous value.
    m_overwrite_sigma = true;

#if defined(WARPX_ZINDEX)
    // The z_hi wall follows the z_hi field boundary condition:
    //   Neumann   -> floating wall (collected charge injected as a sheet)
    //   Dirichlet -> fixed-potential wall (boundary.potential_hi_z, no sheet)
    // The z_hi particle buffer is drained by AccumulateWallCharge either way.
    m_zhi_floating =
        (m_poisson_boundary_handler->hibc[WARPX_ZINDEX] == amrex::LinOpBCType::Neumann);
    amrex::Print() << "LabFrameDriftKineticES: z_hi wall = "
                   << (m_zhi_floating ? "floating (Neumann + charge sheet)"
                                      : "Dirichlet (fixed potential)") << '\n';
#endif

    RestoreWallCharge();
}

amrex::Real LabFrameDriftKineticES::AccumulateWallCharge ()
{
    // The floating wall is an axial (z) concept; RCYLINDER/RSPHERE have no
    // z-axis (WARPX_ZINDEX undefined there), so this is a no-op for them.
#if defined(WARPX_ZINDEX)
    auto & warpx = WarpX::GetInstance();
    auto & bb = warpx.GetParticleBoundaryBuffer();
    auto & mpc = warpx.GetPartContainer();

    // Index of the z_hi boundary in the ParticleBoundaryBuffer (idim*2 + side).
    const int zhi = 2 * WARPX_ZINDEX + 1;

    Real dq = 0.0_rt;
    if (bb.isDefinedForAnySpecies(zhi))
    {
        const std::vector<std::string> species_names = mpc.GetSpeciesNames();
        for (int s = 0; s < mpc.nSpecies(); ++s)
        {
            // The per-species z_hi buffer is only "defined" once particles
            // have actually been collected there. Before that (e.g. the
            // initial space-charge solve, or steps before the first particle
            // reaches the wall) skip it -- it contributes no wall charge.
            auto * buf_ptr = bb.getParticleBufferPointer(species_names[s], zhi);
            if (buf_ptr == nullptr || !buf_ptr->isDefined()) { continue; }

            const Real q = mpc.GetParticleContainer(s).getCharge();

            // Sum the weights of all particles collected at z_hi (the buffer
            // lives in pinned host memory, so a host loop is fine).
            Real wsum = 0.0_rt;
            for (int lev = 0; lev <= buf_ptr->finestLevel(); ++lev) {
                for (auto const& kv : buf_ptr->GetParticles(lev)) {
                    auto const& wcomp = kv.second.GetStructOfArrays().GetRealData(PIdx::w);
                    for (auto const w : wcomp) { wsum += w; }
                }
            }
            dq += q * wsum;
        }
    }
    // Net charge collected this step, summed across MPI ranks.
    ParallelDescriptor::ReduceRealSum(dq);

    m_sigma_wall += dq;

    // Clear the z_hi buffer so the next step only sees newly collected particles.
    bb.clearParticles(zhi);
#endif

    return m_sigma_wall;
}

void LabFrameDriftKineticES::LogWallCharge () const
{
    // Only the I/O rank maintains the state log.
    if (!ParallelDescriptor::IOProcessor()) { return; }

    const int step = WarpX::GetInstance().getistep(0);
    std::ofstream ofs(m_wall_charge_log, std::ios::app);
    if (ofs.is_open()) {
        ofs << step << '\t' << std::setprecision(17) << m_sigma_wall << '\n';
    }
}

void LabFrameDriftKineticES::RestoreWallCharge ()
{
    // Checkpoint persistence for m_sigma_wall. It is a single scalar carried by
    // no registered field, so it is mirrored to a plain-text log -- one
    // "step <tab> value" line per step, written by LogWallCharge(). On restart
    // we recover it from that log; on a fresh run we start the log clean. This
    // keeps the floating wall checkpoint-compatible without modifying core
    // WarpX checkpoint I/O.
    if (ParallelDescriptor::IOProcessor())
    {
        std::string restart;
        {
            const ParmParse pp_amr("amr");
            pp_amr.query("restart", restart);
        }

        if (restart.empty())
        {
            // Fresh run: truncate (start) the state log.
            std::ofstream ofs(m_wall_charge_log, std::ios::trunc);
            m_sigma_wall = 0.0_rt;
        }
        else
        {
            // Restart: parse the checkpoint step from the trailing digits of
            // the restart path (e.g. ".../chk000500" -> 500) and look it up.
            while (!restart.empty() &&
                   (restart.back() == '/' || restart.back() == '\\'))
            {
                restart.pop_back();
            }
            std::size_t p = restart.size();
            while (p > 0 &&
                   std::isdigit(static_cast<unsigned char>(restart[p-1])) != 0)
            {
                --p;
            }

            long chk_step = -1;
            if (p < restart.size()) { chk_step = std::stol(restart.substr(p)); }

            long best_step = -1;
            amrex::Real restored = 0.0_rt;
            if (chk_step >= 0)
            {
                std::ifstream ifs(m_wall_charge_log);
                long s = 0;
                amrex::Real v = 0.0_rt;
                // Most recent log entry at or before the checkpoint step
                // (exact in the common case; tolerates a +-1 step offset in
                // the step <-> checkpoint-name convention).
                while (ifs >> s >> v)
                {
                    if (s <= chk_step && s >= best_step) { best_step = s; restored = v; }
                }
            }

            if (best_step >= 0)
            {
                m_sigma_wall = restored;
                amrex::Print() << "LabFrameDriftKineticES: restored floating-wall "
                    "charge " << m_sigma_wall << " (log step " << best_step
                    << ", checkpoint step " << chk_step << ").\n";
            }
            else
            {
                m_sigma_wall = 0.0_rt;
                amrex::Print() << Utils::TextMsg::Warn(
                    "LabFrameDriftKineticES: could not restore the floating-wall "
                    "charge from '" + m_wall_charge_log + "'; starting it from "
                    "zero. (Expected a 'step value' log from the previous run.)");
            }
        }
    }

    // All ranks receive the value determined by the I/O rank.
    ParallelDescriptor::Bcast(&m_sigma_wall, 1, ParallelDescriptor::IOProcessorNumber());
}

void LabFrameDriftKineticES::ComputeSpaceChargeField (
    ablastr::fields::MultiFabRegister& fields,
    MultiParticleContainer& mpc,
    MultiFluidContainer* mfl,
    int max_level)
{
    ABLASTR_PROFILE("LabFrameDriftKineticES::ComputeSpaceChargeField");

    using ablastr::fields::MultiLevelScalarField;
    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    bool const skip_lev0_coarse_patch = true;

    const MultiLevelScalarField rho_fp = fields.get_mr_levels(FieldType::rho_fp, max_level);
    const MultiLevelScalarField rho_cp = fields.get_mr_levels(FieldType::rho_cp, max_level, skip_lev0_coarse_patch);
    const MultiLevelScalarField phi_fp = fields.get_mr_levels(FieldType::phi_fp, max_level);
    const MultiLevelVectorField Efield_fp = fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);

    // DepositCharge divides out the flux-tube area, so rho_fp holds the 3D
    // charge density n_3D from here on (see FluxTubeAreaScaling.H). The fluid
    // deposition below has no such rescaling, so the two cannot be mixed.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mfl == nullptr,
        "LabFrameDriftKineticES does not support fluid species: the fluid charge "
        "deposition is not rescaled by the flux-tube area A(z).");

    mpc.DepositCharge(rho_fp, 0.0_rt);

    // Apply filter, perform MPI exchange, interpolate across levels
    const Vector<std::unique_ptr<MultiFab> > rho_buf(num_levels);
    auto & warpx = WarpX::GetInstance();
    warpx.SyncRho( rho_fp, rho_cp, amrex::GetVecOfPtrs(rho_buf) );

#ifndef WARPX_DIM_RZ
    for (int lev = 0; lev < num_levels; lev++) {
        warpx.ApplyRhofieldBoundary(lev, rho_fp[lev], PatchType::fine);
    }
#endif

    // Build the flux-tube Poisson source term. The physical equation is
    //     d_z( A eps0 d_z phi ) = -A rho_3D,
    // so the right-hand side carries the area that the deposition divided out.
    // Everything above this line -- filtering, the guard-cell sum, the boundary
    // reflection -- therefore acts on the physical density n_3D; only the MLMG
    // source is extensive. ComputeSigma supplies the matching A on the
    // left-hand side.
    for (int lev = 0; lev < num_levels; lev++) {
        warpx::drift_kinetic::ScaleChargeDensityByFluxTubeArea(rho_fp[lev], lev);
    }

    // Floating right (z_hi) wall: accumulate the charge collected there and
    // inject it as a charge sheet into rho at the boundary node. With a
    // homogeneous Neumann field BC at z_hi this is the dynamic-Neumann
    // (charge-sheet) floating-wall condition.
    const Real sigma_wall = AccumulateWallCharge();
    // Persist the running wall charge so it survives checkpoint/restart.
    LogWallCharge();
#if defined(WARPX_DIM_1D_Z)
    // Floating (Neumann) z_hi wall: inject the collected charge as a sheet at
    // the boundary node. A Dirichlet z_hi wall pins the potential instead, so
    // no sheet is added -- but the wall buffer is still drained above.
    if (m_zhi_floating)
    {
        const Real dz = warpx.Geom(0).CellSize(0);
        const int khi = warpx.Geom(0).Domain().bigEnd(0) + 1; // last (nodal) z index
        const Real rho_wall = sigma_wall / dz;
        MultiFab & rho0 = *rho_fp[0];
        for (MFIter mfi(rho0); mfi.isValid(); ++mfi) {
            const Box& vbx = mfi.validbox();
            if (khi >= vbx.smallEnd(0) && khi <= vbx.bigEnd(0)) {
                Array4<Real> const& r = rho0.array(mfi);
                ParallelFor(Box(IntVect(khi), IntVect(khi)),
                    [=] AMREX_GPU_DEVICE (int i, int /*j*/, int /*k*/) {
                        r(i,0,0,0) += rho_wall;
                    });
            }
        }
    }
#else
    ignore_unused(sigma_wall);
#endif

    // set the boundary potentials appropriately (z_lo grounded Dirichlet;
    // z_hi is a homogeneous Neumann boundary -> left untouched here)
    setPhiBC(phi_fp, warpx.gett_new(0));

    // perform phi calculation
    computePhi(rho_fp, phi_fp, Efield_fp);

    // Compute the electric field. Note that if an EB is used the electric
    // field will be calculated in the computePhi call.
    if (!EB::enabled()) {
        const std::array<Real, 3> beta = {0._rt};
        computeE( Efield_fp, phi_fp, beta );
    }
}

void LabFrameDriftKineticES::ComputeSigma ( MultiFab& sigma ) const
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const int lev = 0;
    const Real C_SI = m_effective_potential_factor;

    auto & warpx = WarpX::GetInstance();

    // Effective-potential semi-implicit dressing eps_SI (Barnes 2021):
    // eps_SI = 1 + sum_species C_SI * (w_p * dt)^2 / 4, temporally filtered as
    //     eps^n = (1 - a) eps^{n-1} + a eps_instantaneous.
    // The decay is applied to the stored value here and the vacuum term is
    // added back as `a` after the species loop, so that the filter acts on the
    // dressing alone and the 1 stays exact.
    MultiFab & eps_SI = *warpx.m_fields.get(FieldType::effective_potential_sigma, lev);

    Real time_filter_param = m_time_filter_param;
    if (m_overwrite_sigma) {
        // No history yet: take the instantaneous value.
        time_filter_param = 1._rt;
        m_overwrite_sigma = false;
    }
    eps_SI.mult(1._rt - time_filter_param, 0);

    if (C_SI > 0._rt)
    {
        // sigma is a cell-centered array
        amrex::GpuArray<int, 3> const cell_centered = {0, 0, 0};
        amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};
#if defined(WARPX_DIM_3D)
        amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::GpuArray<int, 3> const nodal = {1, 1, 0};
#else
        amrex::GpuArray<int, 3> const nodal = {1, 0, 0};
#endif

        auto& mypc = warpx.GetPartContainer();

        const auto mult_factor = (
            C_SI * warpx.getdt(lev) * warpx.getdt(lev) / (4._rt * PhysConst::epsilon_0)
        );
        const Real density_floor = m_density_floor;

        for (auto const& pc : mypc) {
            if (pc->do_not_deposit) { continue; }

            // GetChargeDensity divides out the flux-tube area, so this is the
            // 3D density n_3D and the dressing below is the true w_p of the
            // species -- not the value reduced by A(z) at the mirror throat.
            auto rho = pc->GetChargeDensity(lev, true);
            warpx.ApplyFilterandSumBoundaryRho(lev, lev, *rho, 0, rho->nComp());

            auto const q = std::abs(pc->getCharge());
            auto const mult_factor_pc = mult_factor * q / pc->getMass();
            // Charge density corresponding to the density floor, so that
            // near-vacuum cells keep a dressing rather than falling back to an
            // undressed (explicit-stability-limited) sigma.
            const Real rho_floor = static_cast<Real>(density_floor * q);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for ( MFIter mfi(eps_SI, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
                Array4<Real> const& eps_arr = eps_SI.array(mfi);
                Array4<Real const> const& rho_arr = rho->const_array(mfi);

                amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    auto const rho_cc = std::max(
                        rho_floor,
                        std::abs(ablastr::coarsen::sample::Interp(
                            rho_arr, nodal, cell_centered, coarsen, i, j, k, 0
                        ))
                    );
                    eps_arr(i, j, k, 0) += time_filter_param * mult_factor_pc * rho_cc;
                });
            }
        }
    }

    // Restore the vacuum term. Combined with the decay above this gives
    // eps^n = 1 + [(1 - a) D^{n-1} + a D^n], i.e. only the dressing D is
    // filtered while the 1 is reproduced exactly.
    eps_SI.plus(time_filter_param, 0);

    // sigma = A(z) * eps_SI. Guard cells are left at 1 (as before): the A
    // scaling below covers the valid region only, which is what MLMG reads.
    sigma.setVal(1.0_rt);
    MultiFab::Copy(sigma, eps_SI, 0, 0, 1, 0);

    // Fold in the magnetic flux-tube area A(z) = B_ref/B(z): sigma = A * eps_SI.
    // A is computed here from the *registered* field Bfield_fp_external (rather
    // than from a stored MultiFab) so it always uses the current distribution
    // mapping -- a stored MultiFab would go stale when load balancing changes
    // the box-to-rank assignment.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx.m_fields.has(FieldType::Bfield_fp_external, Direction{2}, lev),
        "LabFrameDriftKineticES requires an external grid B field "
        "(set it via picmi.AnalyticInitialField / B_ext_grid_init_style).");

    const MultiFab & Bz = *warpx.m_fields.get(FieldType::Bfield_fp_external, Direction{2}, lev);
    const Real Bref = m_reference_B;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(sigma, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        Array4<Real>       const& sigma_arr = sigma.array(mfi);
        Array4<Real const> const& Bz_arr    = Bz.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Bz_external is nodal; the cell-centered value is the average of
            // the two surrounding nodes along the axial (z) direction.
#if defined(WARPX_DIM_1D_Z)
            const Real Bc = 0.5_rt * (Bz_arr(i,j,k) + Bz_arr(i+1,j,k));
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            const Real Bc = 0.5_rt * (Bz_arr(i,j,k) + Bz_arr(i,j+1,k));
#else
            const Real Bc = 0.5_rt * (Bz_arr(i,j,k) + Bz_arr(i,j,k+1));
#endif
            const Real A = (Bc > 0._rt) ? Bref / Bc : 1._rt;
            sigma_arr(i,j,k,0) *= A;
        });
    }
}

void LabFrameDriftKineticES::computePhi (
    ablastr::fields::MultiLevelScalarField const& rho,
    ablastr::fields::MultiLevelScalarField const& phi,
    ablastr::fields::MultiLevelVectorField const& efield ) const
{
    auto const& ba = convert(rho[0]->boxArray(), IntVect(AMREX_D_DECL(0,0,0)));
    MultiFab sigma(ba, rho[0]->DistributionMap(), 1, rho[0]->nGrowVect());
    ComputeSigma(sigma);

    computePhi(rho, phi, efield, sigma, self_fields_required_precision,
                self_fields_absolute_tolerance, self_fields_max_iters,
                self_fields_verbosity);
}

void LabFrameDriftKineticES::computePhi (
    ablastr::fields::MultiLevelScalarField const& rho,
    ablastr::fields::MultiLevelScalarField const& phi,
    ablastr::fields::MultiLevelVectorField const& efield,
    amrex::MultiFab const& sigma,
    amrex::Real required_precision,
    amrex::Real absolute_tolerance,
    int max_iters,
    int verbosity
) const
{
    amrex::Vector<amrex::MultiFab *> sorted_rho;
    amrex::Vector<amrex::MultiFab *> sorted_phi;
    for (int lev = 0; lev < num_levels; ++lev) {
        sorted_rho.emplace_back(rho[lev]);
        sorted_phi.emplace_back(phi[lev]);
    }

    auto & warpx = WarpX::GetInstance();

    std::optional<EBCalcEfromPhiPerLevel> post_phi_calculation;
#ifdef AMREX_USE_EB
    std::optional<amrex::Vector<amrex::EBFArrayBoxFactory const *> > eb_farray_box_factory;
#else
    std::optional<amrex::Vector<amrex::FArrayBoxFactory const *> > const eb_farray_box_factory;
#endif
    if (EB::enabled())
    {
        amrex::Vector<
            amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM>
        > e_field;
        for (int lev = 0; lev < num_levels; ++lev) {
            e_field.push_back(
#if defined(WARPX_DIM_1D_Z)
                amrex::Array<amrex::MultiFab*, 1>{
                    efield[lev][2]
                }
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                amrex::Array<amrex::MultiFab*, 1>{
                    efield[lev][0]
                }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::Array<amrex::MultiFab*, 2>{
                    efield[lev][0], efield[lev][2]
                }
#elif defined(WARPX_DIM_3D)
                amrex::Array<amrex::MultiFab *, 3>{
                    efield[lev][0], efield[lev][1], efield[lev][2]
                }
#endif
            );
        }
        post_phi_calculation = EBCalcEfromPhiPerLevel(e_field);

#ifdef AMREX_USE_EB
        amrex::Vector<
            amrex::EBFArrayBoxFactory const *
        > factories;
        for (int lev = 0; lev < num_levels; ++lev) {
            factories.push_back(&warpx.fieldEBFactory(lev));
        }
        eb_farray_box_factory = factories;
#endif
    }

    ablastr::fields::computeEffectivePotentialPhi(
        sorted_rho,
        sorted_phi,
        sigma,
        required_precision,
        absolute_tolerance,
        max_iters,
        verbosity,
        warpx.Geom(),
        warpx.DistributionMap(),
        warpx.boxArray(),
        WarpX::grid_type,
        false,
        EB::enabled(),
        WarpX::do_single_precision_comms,
        warpx.refRatio(),
        post_phi_calculation,
        *m_poisson_boundary_handler,
        warpx.gett_new(0),
        eb_farray_box_factory
    );
}
