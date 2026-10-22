/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_GATHERAXIALFIELDANDGRADIENT_H_
#define WARPX_PARTICLES_PUSHER_GATHERAXIALFIELDANDGRADIENT_H_

#include <AMReX_Array4.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

/** \file
 * Energy-conserving gather of a nodal axial field and its gradient.
 *
 * The drift-kinetic mirror force and the magnetic moment are not independent.
 * Over one step the re-slaving of u_perp to mu = u_perp^2/(2B) changes the
 * perpendicular energy by exactly mu [ B_p(z^{n+1}) - B_p(z^n) ], where B_p is
 * whatever the gather produces; the parallel kick removes mu (dB/dz)_p dz from
 * the parallel energy. Total energy is conserved only if the gathered gradient
 * is the exact derivative of the gathered field:
 *
 *     (dB/dz)_p = d/dz [ sum_j S_j(z) B_j ] = sum_j S'_j(z) B_j.
 *
 * Differencing B on the grid first and then interpolating the result -- the
 * obvious implementation, and what this replaces -- gives a gradient that is
 * second-order accurate pointwise but is not the derivative of the B_p the
 * re-slaving transports. That does not make the energy drift: any 1D force
 * field is conservative, so the step conserves mu G + u^2/2 for the
 * antiderivative G of the gathered gradient, and the physical energy
 * mu B_p + u^2/2 just oscillates by mu (B_p - G) over the bounce. The real
 * cost is that the particle then mirrors off G rather than off B_p, so the
 * effective mirror ratio and the loss-cone boundary are those of a field
 * nobody asked for. Evaluating both sums from the same shape S makes G and
 * B_p the same function, by construction rather than by a cancellation that
 * happens to be clean for a symmetric trapped orbit.
 *
 * Order 2 (S the quadratic B-spline, as algo.particle_shape = 2) makes the
 * gradient piecewise linear and continuous. Measured against the analytic
 * dB/dz of a mirror well of ratio 10 on a 1024-cell grid, it is about twice
 * as accurate pointwise as the grid-differenced version (9.1e-6 vs 1.8e-5
 * RMS relative) and halves the bounded energy excursion of a trapped orbit
 * (4.4e-6 vs 1.0e-5), with no secular drift in either case. It is the default.
 *
 * Order 1 (S piecewise linear) makes the gradient piecewise constant. It is
 * exactly conservative for any step that stays inside one cell, but a step
 * that crosses a node picks up mu (slope_new - slope_old) * overshoot, whose
 * sign follows B'' rather than averaging away, and it is only first-order
 * accurate pointwise (2.6e-3 RMS relative in the same test, and the only one
 * of the three with a measurable secular drift, 1.4e-8 per bounce). Keep it
 * for comparison, not for production.
 */

namespace warpx::drift_kinetic
{

/** \brief Node indices the order-`Order` axial stencil reads at `zn`.
 *
 * \param[in]  zn   particle position in grid units, (z - z_node0)/dz
 * \param[out] imin lowest node index read
 * \param[out] imax highest node index read
 */
template <int Order>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void axialStencilRange (const amrex::Real zn, int& imin, int& imax) noexcept
{
    using namespace amrex::literals;

    if constexpr (Order == 2) {
        // Centred on the nearest node.
        const int i = static_cast<int>(amrex::Math::floor(zn + 0.5_rt));
        imin = i - 1;
        imax = i + 1;
    } else {
        // Spans the cell the particle is in.
        const int i = static_cast<int>(amrex::Math::floor(zn));
        imin = i;
        imax = i + 1;
    }
}

/** \brief Gather a nodal axial field and its exact axial derivative.
 *
 * `f_p` is the order-`Order` B-spline interpolation of `f` at the particle and
 * `dfdz_p` is the analytic z-derivative of that same interpolation, so
 * `dfdz_p` integrates to exactly the change in `f_p` as long as the stencil
 * does not move -- within one cell for order 1, between the same three nodes
 * for order 2.
 *
 * \param[in]  zp     particle axial position
 * \param[in]  f      Array4 of the nodal field (1D: j = k = 0)
 * \param[in]  dzi    inverse cell size 1/dz
 * \param[in]  zlo    physical position of node index 0 of `f`
 * \param[out] f_p    interpolated field at the particle
 * \param[out] dfdz_p derivative of the interpolant at the particle
 */
template <int Order>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void doGatherAxialFieldAndGradient (
    const amrex::ParticleReal zp,
    amrex::Array4<const amrex::Real> const& f,
    const amrex::Real dzi, const amrex::Real zlo,
    amrex::Real& f_p, amrex::Real& dfdz_p) noexcept
{
    using namespace amrex::literals;

    const amrex::Real zn = (static_cast<amrex::Real>(zp) - zlo) * dzi;

    if constexpr (Order == 2)
    {
        // Quadratic B-spline about the nearest node, xi in [-1/2, 1/2]:
        //   S_{-1} = (1/2)(1/2 - xi)^2,  S_0 = 3/4 - xi^2,  S_{+1} = (1/2)(1/2 + xi)^2
        // The derivatives sum to zero, so a uniform field gives no force, and
        // they reproduce the slope of a linear field exactly.
        const int i = static_cast<int>(amrex::Math::floor(zn + 0.5_rt));
        const amrex::Real xi = zn - static_cast<amrex::Real>(i);

        const amrex::Real sm = 0.5_rt * (0.5_rt - xi) * (0.5_rt - xi);
        const amrex::Real s0 = 0.75_rt - xi * xi;
        const amrex::Real sp = 0.5_rt * (0.5_rt + xi) * (0.5_rt + xi);

        const amrex::Real fm = f(i-1, 0, 0);
        const amrex::Real f0 = f(i  , 0, 0);
        const amrex::Real fp = f(i+1, 0, 0);

        f_p    = sm * fm + s0 * f0 + sp * fp;
        dfdz_p = ((xi - 0.5_rt) * fm - 2.0_rt * xi * f0 + (xi + 0.5_rt) * fp) * dzi;
    }
    else
    {
        // Linear shape over the cell: the derivative is the one-sided
        // difference, constant across the cell.
        const int i = static_cast<int>(amrex::Math::floor(zn));
        const amrex::Real w1 = zn - static_cast<amrex::Real>(i);

        const amrex::Real f0 = f(i  , 0, 0);
        const amrex::Real f1 = f(i+1, 0, 0);

        f_p    = (1.0_rt - w1) * f0 + w1 * f1;
        dfdz_p = (f1 - f0) * dzi;
    }
}

/** \brief Gather only the field, with the same interpolant as
 *         doGatherAxialFieldAndGradient.
 *
 * Used by the magnetic-moment re-slaving, which has to see exactly the B_p
 * whose derivative drove the parallel kick -- a different interpolation order
 * there would reintroduce the mismatch this header exists to remove.
 */
template <int Order>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real doGatherAxialField (
    const amrex::ParticleReal zp,
    amrex::Array4<const amrex::Real> const& f,
    const amrex::Real dzi, const amrex::Real zlo) noexcept
{
    amrex::Real f_p = amrex::Real(0.);
    amrex::Real dfdz_p = amrex::Real(0.);
    doGatherAxialFieldAndGradient<Order>(zp, f, dzi, zlo, f_p, dfdz_p);
    return f_p;
}

} // namespace warpx::drift_kinetic

#endif // WARPX_PARTICLES_PUSHER_GATHERAXIALFIELDANDGRADIENT_H_
