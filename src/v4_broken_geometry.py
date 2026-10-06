"""
V4.2 broken radial dust distribution.

This module changes ONLY the spatial weighting of an already physical radial
dust response. Thermal equilibrium, Q_abs, sublimation, WISE band integration,
and observer delays remain unchanged.

The 3-D number density is continuous at r_break:

 n(r) ∝ r^(-p_inner),                              r <= r_break
 n(r) ∝ r_break^(p_outer-p_inner) r^(-p_outer),   r >  r_break

Because clumps are sampled in volume, the radial probability density is
P(r) ∝ r^2 n(r).

Sampling is inverse-CDF on a dense deterministic grid. Angular sampling and
inclination use the same conventions as the V4 geometry.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

PC_TO_LIGHT_DAYS = 3.261563777 * 365.25


def broken_number_density(r_pc, r_break_pc, p_inner, p_outer):
    r = np.asarray(r_pc, dtype=float)
    rb = float(r_break_pc)
    if rb <= 0:
        raise ValueError("r_break_pc must be positive.")
    inner = r ** (-float(p_inner))
    continuity = rb ** (float(p_outer) - float(p_inner))
    outer = continuity * r ** (-float(p_outer))
    return np.where(r <= rb, inner, outer)


def sample_broken_radii(
    n_clumps, r_in_pc, r_break_pc, r_out_pc,
    p_inner, p_outer, rng,
):
    rin, rb, rout = map(float, (r_in_pc, r_break_pc, r_out_pc))
    if not (0 < rin < rb < rout):
        raise ValueError("Require 0 < Rin < Rbreak < Rout.")
    grid = np.geomspace(rin, rout, 20000)
    pdf = grid**2 * broken_number_density(grid, rb, p_inner, p_outer)

    # Numerical CDF using trapezoids.
    dr = np.diff(grid)
    area = 0.5 * (pdf[:-1] + pdf[1:]) * dr
    cdf = np.r_[0.0, np.cumsum(area)]
    cdf /= cdf[-1]

    u = rng.random(int(n_clumps))
    return np.interp(u, cdf, grid)


def generate_broken_clumpy_geometry(
    n_clumps,
    r_in_pc,
    r_break_pc,
    r_out_pc,
    p_inner,
    p_outer,
    theta_open_deg,
    inclination_deg,
    seed=12345,
):
    """
    Generate a 3-D clumpy torus with a continuous broken radial density law.

    theta_open_deg follows V4: polar half-opening angle measured from the axis.
    Dust occupies theta_open <= theta <= pi-theta_open.
    Observer is +z after inclination rotation.
    """
    rng = np.random.default_rng(int(seed))
    r = sample_broken_radii(
        n_clumps, r_in_pc, r_break_pc, r_out_pc,
        p_inner, p_outer, rng,
    )

    th0 = np.deg2rad(float(theta_open_deg))
    if not (0 <= th0 < np.pi/2):
        raise ValueError("theta_open_deg must be in [0, 90).")

    # Uniform solid angle in the allowed equatorial band.
    cmax = np.cos(th0)
    cos_theta = rng.uniform(-cmax, cmax, int(n_clumps))
    sin_theta = np.sqrt(np.maximum(0.0, 1.0-cos_theta**2))
    phi = rng.uniform(0.0, 2*np.pi, int(n_clumps))

    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta

    # Rotate torus by inclination about x-axis; observer remains +z.
    inc = np.deg2rad(float(inclination_deg))
    y_obs = y*np.cos(inc) - z*np.sin(inc)
    z_obs = y*np.sin(inc) + z*np.cos(inc)
    x_obs = x

    echo_delay_days = (r-z_obs) * PC_TO_LIGHT_DAYS

    return pd.DataFrame({
        "r_pc": r,
        "x_obs_pc": x_obs,
        "y_obs_pc": y_obs,
        "z_obs_pc": z_obs,
        "echo_delay_rest_days": echo_delay_days,
    })
