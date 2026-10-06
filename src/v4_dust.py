"""
v4_dust.py
===========

3-D clumpy-dust geometry for the physical UGC 11487 V4 echo model.

This module contains NO WISE data and NO fitted dust temperatures.

The dust distribution is described by:
    R_in, R_out
    radial number-density slope p
    angular opening angle
    inclination

Each Monte-Carlo clump has a physical 3-D position and therefore:
    - source -> clump propagation time
    - clump -> observer echo delay
    - physical radius

Temperature and sublimation are calculated later from the
time-dependent illuminating luminosity.
"""

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================

C_CGS = 2.99792458e10
DAY_S = 86400.0
PC_CGS = 3.085677581491367e18

LIGHT_DAY_PC = C_CGS * DAY_S / PC_CGS


# ============================================================
# Radial sampling
# ============================================================

def sample_radius_powerlaw(
    n_clumps,
    r_in_pc,
    r_out_pc,
    p,
    rng,
):
    """
    Sample radii for a 3-D number density

        n(r) proportional to r^(-p).

    The probability of finding a clump in shell dr is

        dP proportional to n(r) * 4*pi*r^2 dr
           proportional to r^(2-p) dr.

    Therefore this function samples the physically relevant
    volume-weighted radial distribution.

    Parameters
    ----------
    n_clumps : int
    r_in_pc, r_out_pc : float
    p : float
        3-D number-density power-law index.
    rng : numpy.random.Generator
    """

    n_clumps = int(n_clumps)
    r_in = float(r_in_pc)
    r_out = float(r_out_pc)
    p = float(p)

    if n_clumps <= 0:
        raise ValueError("n_clumps must be positive.")

    if r_in <= 0:
        raise ValueError("r_in_pc must be > 0.")

    if r_out <= r_in:
        raise ValueError("r_out_pc must be > r_in_pc.")

    u = rng.random(n_clumps)

    exponent = 3.0 - p

    # Special logarithmic case p = 3.
    if abs(exponent) < 1.0e-10:
        return r_in * np.exp(
            u * np.log(r_out / r_in)
        )

    a = r_in**exponent
    b = r_out**exponent

    return (
        a + u * (b - a)
    ) ** (1.0 / exponent)


# ============================================================
# Angular geometry
# ============================================================

def sample_torus_angles(
    n_clumps,
    theta_open_deg,
    rng,
):
    """
    Sample an axisymmetric dusty distribution outside two
    polar dust-free cones.

    theta is measured from the symmetry axis.

    theta_open_deg is the polar half-opening angle:
        theta_open < theta < pi - theta_open

    Sampling is uniform in solid angle inside the dusty region.
    """

    n_clumps = int(n_clumps)
    theta_open = np.deg2rad(
        float(theta_open_deg)
    )

    if not (
        0.0 <= theta_open < 0.5 * np.pi
    ):
        raise ValueError(
            "theta_open_deg must satisfy 0 <= theta_open < 90."
        )

    mu_max = np.cos(theta_open)

    # Uniform solid-angle sampling means uniform cos(theta).
    mu = rng.uniform(
        -mu_max,
        mu_max,
        size=n_clumps,
    )

    theta = np.arccos(mu)

    phi = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=n_clumps,
    )

    return theta, phi


# ============================================================
# Coordinate transformation
# ============================================================

def intrinsic_cartesian(
    radius_pc,
    theta,
    phi,
):
    """
    Cartesian coordinates in the intrinsic torus frame.
    Symmetry axis = z.
    """

    r = np.asarray(radius_pc, dtype=float)

    st = np.sin(theta)

    x = r * st * np.cos(phi)
    y = r * st * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def rotate_for_inclination(
    x,
    y,
    z,
    inclination_deg,
):
    """
    Rotate torus about the x-axis.

    inclination = 0 deg:
        observer looks along symmetry axis.

    inclination = 90 deg:
        edge-on.

    After rotation the observer is placed at +z_obs infinity.
    """

    inc = np.deg2rad(
        float(inclination_deg)
    )

    ci = np.cos(inc)
    si = np.sin(inc)

    x_obs = x

    y_obs = y * ci - z * si

    z_obs = y * si + z * ci

    return x_obs, y_obs, z_obs


# ============================================================
# Light-travel geometry
# ============================================================

def source_to_clump_days(radius_pc):
    """
    Source -> clump propagation time r/c.
    """

    return (
        np.asarray(radius_pc, dtype=float)
        / LIGHT_DAY_PC
    )


def echo_delay_days(
    radius_pc,
    z_observer_pc,
):
    """
    Observed echo delay relative to direct source light.

    Observer lies at +z infinity.

        tau_echo = (r - z_obs) / c

                 = r/c * (1 - cos(alpha))

    where alpha is the angle between the clump position vector
    and the observer line of sight.

    This naturally gives:
        near side  -> short delay
        plane sky  -> r/c
        far side   -> up to 2r/c
    """

    r = np.asarray(radius_pc, dtype=float)
    z = np.asarray(z_observer_pc, dtype=float)

    delay_pc = r - z

    # Numerical round-off protection only.
    delay_pc = np.maximum(
        delay_pc,
        0.0,
    )

    return delay_pc / LIGHT_DAY_PC


# ============================================================
# Geometry construction
# ============================================================

def generate_clumpy_geometry(
    n_clumps,
    r_in_pc,
    r_out_pc,
    p,
    theta_open_deg=70.0,
    inclination_deg=55.0,
    seed=12345,
):
    """
    Generate one Monte-Carlo realization of a continuous
    clumpy dusty structure.

    No thermal zones are introduced.

    Returns
    -------
    pandas.DataFrame
    """

    rng = np.random.default_rng(
        int(seed)
    )

    radius = sample_radius_powerlaw(
        n_clumps=n_clumps,
        r_in_pc=r_in_pc,
        r_out_pc=r_out_pc,
        p=p,
        rng=rng,
    )

    theta, phi = sample_torus_angles(
        n_clumps=n_clumps,
        theta_open_deg=theta_open_deg,
        rng=rng,
    )

    x, y, z = intrinsic_cartesian(
        radius,
        theta,
        phi,
    )

    xo, yo, zo = rotate_for_inclination(
        x,
        y,
        z,
        inclination_deg,
    )

    source_delay = source_to_clump_days(
        radius
    )

    echo_delay = echo_delay_days(
        radius,
        zo,
    )

    cos_alpha = np.divide(
        zo,
        radius,
        out=np.zeros_like(zo),
        where=radius > 0,
    )

    df = pd.DataFrame({
        "clump_id":
            np.arange(len(radius), dtype=int),

        "r_pc":
            radius,

        "theta_rad":
            theta,

        "phi_rad":
            phi,

        "x_pc":
            x,

        "y_pc":
            y,

        "z_pc":
            z,

        "x_obs_pc":
            xo,

        "y_obs_pc":
            yo,

        "z_obs_pc":
            zo,

        "cos_alpha":
            cos_alpha,

        "source_to_clump_days":
            source_delay,

        "echo_delay_days":
            echo_delay,

        # Initially every clump exists.
        # Sublimation module will update this dynamically.
        "initially_alive":
            np.ones(len(radius), dtype=bool),
    })

    return df


# ============================================================
# Geometry diagnostics
# ============================================================

def geometry_diagnostics(df):
    """
    Numerical/physical diagnostics for one clump realization.
    """

    r = df["r_pc"].to_numpy(float)
    tau_src = df[
        "source_to_clump_days"
    ].to_numpy(float)

    tau_echo = df[
        "echo_delay_days"
    ].to_numpy(float)

    mu = df[
        "cos_alpha"
    ].to_numpy(float)

    diag = {
        "n_clumps":
            int(len(df)),

        "r_min_pc":
            float(np.min(r)),

        "r_median_pc":
            float(np.median(r)),

        "r_mean_pc":
            float(np.mean(r)),

        "r_max_pc":
            float(np.max(r)),

        "source_delay_min_days":
            float(np.min(tau_src)),

        "source_delay_median_days":
            float(np.median(tau_src)),

        "source_delay_max_days":
            float(np.max(tau_src)),

        "echo_delay_min_days":
            float(np.min(tau_echo)),

        "echo_delay_median_days":
            float(np.median(tau_echo)),

        "echo_delay_max_days":
            float(np.max(tau_echo)),

        "cos_alpha_min":
            float(np.min(mu)),

        "cos_alpha_max":
            float(np.max(mu)),
    }

    return diag


def validate_geometry(df):
    """
    Sanity checks independent of WISE data.
    """

    messages = []

    r = df["r_pc"].to_numpy(float)
    tau_src = df[
        "source_to_clump_days"
    ].to_numpy(float)

    tau_echo = df[
        "echo_delay_days"
    ].to_numpy(float)

    mu = df[
        "cos_alpha"
    ].to_numpy(float)

    if np.any(~np.isfinite(r)):
        messages.append(
            "ERROR: non-finite clump radii."
        )

    if np.any(r <= 0):
        messages.append(
            "ERROR: non-positive clump radius."
        )

    if np.any(tau_src <= 0):
        messages.append(
            "ERROR: non-positive source-to-clump delay."
        )

    if np.any(tau_echo < 0):
        messages.append(
            "ERROR: negative observed echo delay."
        )

    if np.any(mu < -1.0 - 1e-12) or \
       np.any(mu > 1.0 + 1e-12):
        messages.append(
            "ERROR: invalid cos(alpha)."
        )

    # Fundamental geometric bound:
    # 0 <= tau_echo <= 2r/c.
    if np.any(
        tau_echo >
        2.0 * tau_src + 1e-8
    ):
        messages.append(
            "ERROR: echo delay exceeds 2r/c."
        )

    if not messages:
        messages.append(
            "PASS: geometry sanity checks passed."
        )

    return messages