/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QUASINEUTRALBOLTZMANNES_H_
#define WARPX_QUASINEUTRALBOLTZMANNES_H_

#include "ElectrostaticSolver.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_REAL.H>

/**
 * \brief Quasineutral electrostatic solver with Boltzmann electrons.
 *
 * There is no Poisson solve. Electrons are not represented by particles; they
 * are assumed massless, isothermal and in equilibrium along the field line, so
 *
 *     n_e = n_ref exp(e phi / T_e),
 *
 * and quasineutrality n_e = n_i (summed over ion charge states) then gives the
 * potential algebraically from the deposited ion density:
 *
 *     phi(z) = (T_e/e) ln( n_i(z) / n_ref ).
 *
 * E is obtained afterwards as the centred difference of phi, so the force on
 * an ion is the exact discrete gradient of the discrete potential. That keeps
 * the leapfrog symplectic; forming E directly from -(T_e/e) grad(n)/n would
 * not be the gradient of anything discrete and would drift in energy, besides
 * dividing by a density that is small and noisy exactly where it matters.
 *
 * In the drift-kinetic flux tube the deposited density is already divided by
 * the tube area A(z) = B_ref/B(z), so n_i here is the physical 3D density and
 *
 *     phi = (T_e/e) ln(rho_1D/(e n_ref)) - (T_e/e) ln A(z),
 *
 * i.e. the geometry contributes a static barrier -(T_e/e) ln A. For a mirror
 * ratio of 10 that is 2.3 T_e/e at the throat: the ambipolar barrier of the
 * mirror comes out of the area normalisation, and is absent entirely if the
 * charge-per-length is used in place of the 3D density.
 *
 * Limitations, both inherent to the model rather than this implementation:
 *  - Quasineutrality fails within a Debye length of a wall, so there is no
 *    sheath. A wall boundary condition has to be supplied separately if the
 *    sheath matters.
 *  - phi is fixed by the density, so Dirichlet potentials cannot be imposed;
 *    only differences of phi are meaningful, and n_ref is a pure offset.
 */
class QuasineutralBoltzmannES final : public ElectrostaticSolver
{
public:

    QuasineutralBoltzmannES (int nlevs_max) : ElectrostaticSolver (nlevs_max) {
        ReadParameters();
        ReadBoltzmannParameters();
    }

    void InitData () override;

    void ComputeSpaceChargeField (
        ablastr::fields::MultiFabRegister& fields,
        MultiParticleContainer& mpc,
        MultiFluidContainer* mfl,
        int max_level) override;

    /** \brief Fill phi = (T_e/e) ln(n_i/n_ref) from the deposited charge
     *         density, applying the density floor.
     *
     * This is an implementation detail and would otherwise be private, but it
     * contains an extended __device__ lambda and nvcc does not allow those in
     * a function with private or protected access. The other solvers keep
     * their kernel-bearing helpers public for the same reason.
     *
     * \param[in]  rho charge density (nodal), already area-normalised
     * \param[out] phi potential
     * \param[in]  lev refinement level
     */
    void ComputePhiFromDensity (
        amrex::MultiFab const & rho, amrex::MultiFab & phi, int lev ) const;

private:

    /** \brief Read warpx.boltzmann_* input parameters. */
    void ReadBoltzmannParameters ();

    /** Electron temperature [eV]. */
    amrex::Real m_Te = 1.0;

    /** Reference density [m^-3] in n_e = n_ref exp(e phi / T_e). Only shifts
     *  phi by a constant, so it does not affect E; it fixes the zero point. */
    amrex::Real m_n_ref = 1.0e18;

    /** Floor on the ion density [m^-3] used in the logarithm, so that phi
     *  stays finite in vacuum and bounded where the density is noisy. */
    amrex::Real m_density_floor = 1.0e10;
};

#endif // WARPX_QUASINEUTRALBOLTZMANNES_H_
