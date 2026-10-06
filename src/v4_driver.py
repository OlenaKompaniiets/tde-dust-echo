"""
v4_driver.py
============

Physical TDE luminosity-driver construction for the UGC 11487
dust-echo V4 model.

Principles
----------
1. STARS fallback curves are treated in their physical units:
       time       : days
       fallback   : Msun / yr

2. The STARS library reference black-hole mass is 1e6 Msun.
   Curves are scaled to the adopted black-hole mass using

       t      -> t * sqrt(M_BH / M_BH,ref)
       Mdot   -> Mdot / sqrt(M_BH / M_BH,ref)

   so that the integrated fallback mass is conserved.

3. No normalization to an imposed total radiated energy is performed.

4. Accretion can differ from fallback through a causal exponential
   viscous response and an optional circularisation delay.

5. Bolometric luminosity is calculated from

       L_bol = eta_rad * Mdot_acc * c^2.

6. This module contains no WISE data and no dust fitting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import lfilter


# ============================================================
# Physical constants
# ============================================================

C_CGS = 2.99792458e10          # cm s^-1
MSUN_CGS = 1.98847e33          # g
DAY_S = 86400.0
YEAR_S = 365.25 * DAY_S

STARS_REFERENCE_BH_MSUN = 1.0e6


# ============================================================
# Basic physical functions
# ============================================================

def eddington_luminosity(M_BH_Msun):
    """
    Eddington luminosity [erg/s].
    """
    return 1.26e38 * float(M_BH_Msun)


def mdot_msunyr_to_gs(mdot_msun_yr):
    """
    Convert Msun/yr -> g/s.
    """
    return (
        np.asarray(mdot_msun_yr, dtype=float)
        * MSUN_CGS
        / YEAR_S
    )


def luminosity_from_mdot(mdot_msun_yr, eta_rad):
    """
    Bolometric luminosity from accretion rate:

        L = eta * Mdot * c^2

    Returns erg/s.
    """
    return (
        float(eta_rad)
        * mdot_msunyr_to_gs(mdot_msun_yr)
        * C_CGS**2
    )


# ============================================================
# STARS input
# ============================================================

def read_stars_file(path):
    """
    Read a STARS fallback curve.

    Expected physical columns:
        column 1 : time [day]
        column 2 : fallback rate [Msun/yr]

    Comment and blank lines are ignored.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"STARS file not found: {path}")

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:

            s = line.strip()

            if not s or s.startswith("#"):
                continue

            parts = s.replace(",", " ").split()

            if len(parts) < 2:
                continue

            try:
                t = float(parts[0])
                mdot = float(parts[1])
            except ValueError:
                continue

            if np.isfinite(t) and np.isfinite(mdot):
                rows.append((t, mdot))

    if len(rows) < 5:
        raise ValueError(
            f"Could not parse enough numerical rows from {path}"
        )

    arr = np.asarray(rows, dtype=float)

    t_days = arr[:, 0]
    mdot = arr[:, 1]

    order = np.argsort(t_days)

    t_days = t_days[order]
    mdot = mdot[order]

    # Fallback rate cannot be negative.
    mdot = np.clip(mdot, 0.0, None)

    if np.nanmax(mdot) <= 0:
        raise ValueError(
            f"Fallback rate is non-positive in {path}"
        )

    return t_days, mdot


# ============================================================
# STARS BH-mass scaling
# ============================================================

def scale_stars_curve(
    t_days,
    mdot_msun_yr,
    M_BH_Msun,
    M_BH_reference=STARS_REFERENCE_BH_MSUN,
):
    """
    Scale a STARS curve from the reference BH mass to M_BH.

    Scaling:

        t'    = t * sqrt(M_BH / M_ref)

        Mdot' = Mdot / sqrt(M_BH / M_ref)

    The integrated fallback mass is therefore invariant.
    """

    t_days = np.asarray(t_days, dtype=float)
    mdot_msun_yr = np.asarray(
        mdot_msun_yr,
        dtype=float,
    )

    q = float(M_BH_Msun) / float(M_BH_reference)

    if q <= 0:
        raise ValueError("Black-hole mass must be positive.")

    s = np.sqrt(q)

    return (
        t_days * s,
        mdot_msun_yr / s,
    )


# ============================================================
# Integration diagnostics
# ============================================================

def integrated_mass_msun(t_days, mdot_msun_yr):
    """
    Integrate Mdot [Msun/yr] over time [day].

    Returns mass in Msun.
    """

    t_year = np.asarray(t_days, dtype=float) / 365.25

    return float(
        np.trapezoid(
            np.asarray(mdot_msun_yr, dtype=float),
            t_year,
        )
    )


def integrated_energy_erg(t_days, luminosity_erg_s):
    """
    Integrate luminosity over time.

    Returns energy in erg.
    """

    t_seconds = (
        np.asarray(t_days, dtype=float)
        * DAY_S
    )

    return float(
        np.trapezoid(
            np.asarray(luminosity_erg_s, dtype=float),
            t_seconds,
        )
    )


# ============================================================
# Causal accretion response
# ============================================================

def causal_exponential_response(t_grid_days, signal, tau_days):
    """Positive causal discrete exponential with unit infinite-horizon sum.

    Bin-integrated kernel K[k]=(1-exp(-dt/tau))*exp(-k*dt/tau), k>=0.
    No FFT and no tail renormalization. sum(out) <= sum(input); a finite
    output horizon retains a residual tail. Timing error converges with dt.
    """
    t = np.asarray(t_grid_days, float)
    x = np.asarray(signal, float)
    if t.ndim != 1 or t.size < 2 or x.shape != t.shape:
        raise ValueError("Matching 1D time and signal arrays of length >=2 required")
    d = np.diff(t)
    if not np.all(np.isfinite(t)) or np.any(d <= 0) or not np.allclose(d,d[0],rtol=1e-9,atol=1e-10):
        raise ValueError("Time must be finite, uniform and increasing")
    if np.any(~np.isfinite(x)) or np.any(x < 0) or not np.isfinite(tau_days) or tau_days < 0:
        raise ValueError("Finite nonnegative signal and tau required")
    if tau_days == 0:
        return x.copy()
    alpha = np.exp(-d[0]/tau_days)
    beta = -np.expm1(-d[0]/tau_days)
    return lfilter([beta], [1., -alpha], x)


def apply_time_delay(
    t_grid_days,
    signal,
    delay_days,
):
    """
    Apply a positive causal delay.

        output(t) = input(t - delay)
    """

    delay_days = float(delay_days)

    if delay_days < 0:
        raise ValueError(
            "Circularisation delay cannot be negative."
        )

    if delay_days == 0:
        return np.asarray(signal, dtype=float).copy()

    f = interp1d(
        t_grid_days,
        signal,
        bounds_error=False,
        fill_value=0.0,
    )

    return np.clip(
        f(
            np.asarray(t_grid_days, dtype=float)
            - delay_days
        ),
        0.0,
        None,
    )


# ============================================================
# Driver construction
# ============================================================

def build_physical_driver(
    stars_file,
    M_BH_Msun=6.309573e6,
    eta_rad=0.10,
    tau_visc_days=0.0,
    t_circ_days=0.0,
    dt_days=2.0,
    pre_peak_days=1000.0,
    post_fallback_days=6000.0,
):
    """
    Construct a physical fallback/accretion/luminosity driver.

    No imposed total-energy normalization is used.

    Time zero is defined at the peak of the scaled fallback curve.

    Returns
    -------
    df : pandas.DataFrame
        Physical time series.

    diagnostics : dict
        Conservation and energetic diagnostics.
    """

    # --------------------------------------------------------
    # Read STARS
    # --------------------------------------------------------

    t_raw, mdot_raw = read_stars_file(stars_file)

    mass_raw = integrated_mass_msun(
        t_raw,
        mdot_raw,
    )

    # --------------------------------------------------------
    # Scale from 1e6 Msun BH to UGC 11487 BH
    # --------------------------------------------------------

    t_scaled, mdot_scaled = scale_stars_curve(
        t_raw,
        mdot_raw,
        M_BH_Msun=M_BH_Msun,
    )

    mass_scaled_native = integrated_mass_msun(
        t_scaled,
        mdot_scaled,
    )

    # Define t=0 at fallback peak.
    i_peak = int(np.nanargmax(mdot_scaled))
    t_peak = float(t_scaled[i_peak])

    t_scaled = t_scaled - t_peak

    # --------------------------------------------------------
    # Uniform numerical grid
    # --------------------------------------------------------

    dt_days = float(dt_days)

    if dt_days <= 0:
        raise ValueError(
            "dt_days must be positive."
        )

    t_min = min(
        float(np.nanmin(t_scaled)),
        -abs(float(pre_peak_days)),
    )

    # Need a sufficiently long tail for viscous convolution.
    tail_extra = max(
        float(post_fallback_days),
        20.0 * float(tau_visc_days),
        float(t_circ_days) + 1000.0,
    )

    t_max = (
        float(np.nanmax(t_scaled))
        + tail_extra
    )

    t_grid = np.arange(
        t_min,
        t_max + dt_days,
        dt_days,
    )

    # --------------------------------------------------------
    # Interpolate physical fallback rate
    # --------------------------------------------------------

    fb_interp = interp1d(
        t_scaled,
        mdot_scaled,
        bounds_error=False,
        fill_value=0.0,
    )

    mdot_fb = np.clip(
        fb_interp(t_grid),
        0.0,
        None,
    )

    mass_fb_grid = integrated_mass_msun(
        t_grid,
        mdot_fb,
    )

    # --------------------------------------------------------
    # Causal viscous response
    # --------------------------------------------------------

    mdot_acc = causal_exponential_response(
        t_grid,
        mdot_fb,
        tau_visc_days,
    )

    # --------------------------------------------------------
    # Circularisation delay
    # --------------------------------------------------------

    mdot_acc = apply_time_delay(
        t_grid,
        mdot_acc,
        t_circ_days,
    )

    mass_acc_grid = integrated_mass_msun(
        t_grid,
        mdot_acc,
    )

    # --------------------------------------------------------
    # Physical luminosity
    # --------------------------------------------------------

    luminosity = luminosity_from_mdot(
        mdot_acc,
        eta_rad=eta_rad,
    )

    L_edd = eddington_luminosity(
        M_BH_Msun
    )

    E_rad = integrated_energy_erg(
        t_grid,
        luminosity,
    )

    # Expected energy from integrated accreted mass.
    E_expected = (
        float(eta_rad)
        * mass_acc_grid
        * MSUN_CGS
        * C_CGS**2
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    diagnostics = {
        "stars_file": str(Path(stars_file)),
        "M_BH_Msun": float(M_BH_Msun),
        "STARS_reference_M_BH_Msun":
            float(STARS_REFERENCE_BH_MSUN),

        "eta_rad": float(eta_rad),
        "tau_visc_days": float(tau_visc_days),
        "t_circ_days": float(t_circ_days),

        "fallback_mass_raw_Msun":
            mass_raw,

        "fallback_mass_scaled_native_Msun":
            mass_scaled_native,

        "fallback_mass_grid_Msun":
            mass_fb_grid,

        "accreted_mass_grid_Msun":
            mass_acc_grid,

        "mass_scaling_ratio":
            (
                mass_scaled_native / mass_raw
                if mass_raw > 0
                else np.nan
            ),

        "grid_fallback_mass_ratio":
            (
                mass_fb_grid / mass_scaled_native
                if mass_scaled_native > 0
                else np.nan
            ),

        "accretion_mass_ratio":
            (
                mass_acc_grid / mass_fb_grid
                if mass_fb_grid > 0
                else np.nan
            ),

        "mdot_fb_peak_Msun_yr":
            float(np.nanmax(mdot_fb)),

        "mdot_acc_peak_Msun_yr":
            float(np.nanmax(mdot_acc)),

        "L_peak_erg_s":
            float(np.nanmax(luminosity)),

        "L_Edd_erg_s":
            float(L_edd),

        "L_peak_over_L_Edd":
            float(np.nanmax(luminosity) / L_edd),

        "radiated_energy_erg":
            E_rad,

        "expected_eta_Mc2_energy_erg":
            E_expected,

        "energy_conservation_ratio":
            (
                E_rad / E_expected
                if E_expected > 0
                else np.nan
            ),

        "t_grid_min_days":
            float(t_grid.min()),

        "t_grid_max_days":
            float(t_grid.max()),

        "dt_days":
            float(dt_days),
    }

    # --------------------------------------------------------
    # Output table
    # --------------------------------------------------------

    df = pd.DataFrame({
        "t_days": t_grid,
        "mdot_fb_Msun_yr": mdot_fb,
        "mdot_acc_Msun_yr": mdot_acc,
        "L_bol_erg_s": luminosity,
        "L_over_L_Edd": luminosity / L_edd,
    })

    return df, diagnostics


# ============================================================
# Diagnostics validation
# ============================================================

def validate_driver_diagnostics(
    diagnostics,
    mass_tolerance=0.02,
    energy_tolerance=1.0e-6,
):
    """
    Return a list of warnings/errors from physical sanity checks.
    """

    messages = []

    r_scale = diagnostics[
        "mass_scaling_ratio"
    ]

    if not np.isfinite(r_scale):
        messages.append(
            "ERROR: invalid STARS mass-scaling diagnostic."
        )
    elif abs(r_scale - 1.0) > mass_tolerance:
        messages.append(
            "ERROR: STARS BH-mass scaling does not conserve "
            f"fallback mass: ratio={r_scale:.6f}"
        )

    r_grid = diagnostics[
        "grid_fallback_mass_ratio"
    ]

    if not np.isfinite(r_grid):
        messages.append(
            "ERROR: invalid interpolated fallback mass."
        )
    elif abs(r_grid - 1.0) > mass_tolerance:
        messages.append(
            "WARNING: interpolation/grid changed fallback "
            f"mass by {(r_grid - 1.0) * 100:.2f}%."
        )

    r_acc = diagnostics[
        "accretion_mass_ratio"
    ]

    if not np.isfinite(r_acc):
        messages.append(
            "ERROR: invalid accretion mass."
        )
    elif abs(r_acc - 1.0) > mass_tolerance:
        messages.append(
            "WARNING: causal response/grid changed integrated "
            f"mass by {(r_acc - 1.0) * 100:.2f}%."
        )

    r_energy = diagnostics[
        "energy_conservation_ratio"
    ]

    if not np.isfinite(r_energy):
        messages.append(
            "ERROR: invalid radiated-energy diagnostic."
        )
    elif abs(r_energy - 1.0) > energy_tolerance:
        messages.append(
            "ERROR: numerical L=eta*Mdot*c^2 integration "
            f"is inconsistent: ratio={r_energy:.8f}"
        )

    if diagnostics["L_peak_over_L_Edd"] > 1.0:
        messages.append(
            "INFO: intrinsic eta*Mdot*c^2 luminosity is "
            "super-Eddington. No Eddington cap has been "
            "applied."
        )

    if not messages:
        messages.append(
            "PASS: driver conservation checks passed."
        )

    return messages