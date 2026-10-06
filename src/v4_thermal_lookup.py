"""
Fast radiative-equilibrium lookup for the UGC 11487 physical dust-echo V4 model.

The lookup does not introduce a new thermal model.  It tabulates the same
radiative-equilibrium equation used by DustThermalEquilibrium and then uses
interpolation during time-dependent calculations.

For fixed grain size and composition,

    T_eq = T_eq(F_inc),

where

    F_inc = L / (4*pi*r^2).

The source spectral shape is inherited from the supplied
DustThermalEquilibrium instance.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


PC_M = 3.085677581491367e16
ERG_S_TO_W = 1.0e-7


@dataclass
class ThermalLookup:
    grain_radius_micron: np.ndarray
    temperature_grid_K: np.ndarray
    log10_flux_grid_W_m2: np.ndarray
    interpolators: list


def incident_flux_W_m2(luminosity_erg_s, distance_pc):
    """
    Bolometric incident flux L/(4*pi*r^2) in W m^-2.
    """
    L = np.asarray(luminosity_erg_s, dtype=float)
    r = np.asarray(distance_pc, dtype=float)

    if np.any(L < 0.0):
        raise ValueError("luminosity_erg_s must be non-negative.")

    if np.any(r <= 0.0):
        raise ValueError("distance_pc must be strictly positive.")

    L_W = L * ERG_S_TO_W
    r_m = r * PC_M

    return L_W / (4.0 * np.pi * r_m**2)


def build_thermal_lookup(
    thermal_solver,
    grain_radius_micron,
    Tmin_K=2.0,
    Tmax_K=10000.0,
    n_temperature=1600,
):
    """
    Build the inverse radiative-equilibrium relation T(F_inc)
    for every requested grain size.

    Instead of repeatedly root-solving T for millions of (L,r,t)
    combinations, we tabulate the emitted power as a function of
    temperature.

    Radiative equilibrium is

        F_inc * <Qabs>_src
            = 4*pi * integral Qabs(lambda,a) B_lambda(T) dlambda

    so

        F_inc(T)
            = 4*pi * emission_integral(T,a) / <Qabs>_src.

    Parameters
    ----------
    thermal_solver
        DustThermalEquilibrium instance.

    grain_radius_micron : array-like
        Grain radii [micron].

    Tmin_K, Tmax_K : float
        Numerical temperature range of the table.

    n_temperature : int
        Number of logarithmically spaced temperature nodes.

    Returns
    -------
    ThermalLookup
    """

    grains = np.asarray(grain_radius_micron, dtype=float)

    if grains.ndim != 1:
        raise ValueError("grain_radius_micron must be 1D.")

    if np.any(grains <= 0.0):
        raise ValueError("grain_radius_micron must be strictly positive.")

    if Tmin_K <= 0.0 or Tmax_K <= Tmin_K:
        raise ValueError("Invalid temperature limits.")

    if n_temperature < 100:
        raise ValueError("n_temperature is too small for a robust lookup.")

    T_grid = np.geomspace(
        Tmin_K,
        Tmax_K,
        int(n_temperature),
    )

    logF_all = np.empty(
        (grains.size, T_grid.size),
        dtype=float,
    )

    interpolators = []

    for ia, a in enumerate(grains):

        q_source = float(
            thermal_solver.source_mean_qabs(a)
        )

        if not np.isfinite(q_source) or q_source <= 0.0:
            raise RuntimeError(
                f"Invalid source-mean Qabs for a={a} micron: "
                f"{q_source}"
            )

        emission = np.empty_like(T_grid)

        for i, T in enumerate(T_grid):
            emission[i] = thermal_solver.emission_integral(
                temperature_K=float(T),
                radius_micron=float(a),
            )

        flux = (
            4.0
            * np.pi
            * emission
            / q_source
        )

        if np.any(~np.isfinite(flux)):
            raise RuntimeError(
                f"Non-finite equilibrium flux for a={a} micron."
            )

        if np.any(flux <= 0.0):
            raise RuntimeError(
                f"Non-positive equilibrium flux for a={a} micron."
            )

        # Numerical integration should give a strictly increasing relation.
        # Enforce monotonicity only at machine-precision level.
        flux = np.maximum.accumulate(flux)

        if np.any(np.diff(flux) <= 0.0):
            raise RuntimeError(
                f"Equilibrium flux grid is not strictly increasing "
                f"for a={a} micron."
            )

        logF = np.log10(flux)
        logT = np.log10(T_grid)

        logF_all[ia] = logF

        interpolators.append(
            PchipInterpolator(
                logF,
                logT,
                extrapolate=False,
            )
        )

    return ThermalLookup(
        grain_radius_micron=grains,
        temperature_grid_K=T_grid,
        log10_flux_grid_W_m2=logF_all,
        interpolators=interpolators,
    )


def lookup_temperature_from_flux(lookup, incident_flux, grain_index):
    """Inverse equilibrium table, with explicitly separated boundary states.

    F=0 and 0<F<F(Tmin) return 0: unresolved COLD/MIR-dark convention,
    not sublimation or a measured physical temperature. Tmin defaults to 2 K.
    F>F(Tmax) returns +inf; invalid inputs raise instead of destroying dust.
    """
    F = np.asarray(incident_flux, float)
    if np.any(~np.isfinite(F)) or np.any(F < 0):
        raise ValueError("Incident flux must be finite and nonnegative")
    if not 0 <= grain_index < len(lookup.interpolators):
        raise IndexError("grain_index outside lookup table")
    fmin, fmax = 10.**lookup.log10_flux_grid_W_m2[grain_index, [0,-1]]
    out = np.zeros_like(F)
    inside = (F >= fmin) & (F <= fmax)
    out[inside] = 10.**lookup.interpolators[grain_index](np.log10(F[inside]))
    out[F > fmax] = np.inf
    return float(out) if out.ndim == 0 else out


def direct_temperature_from_flux(
    thermal_solver,
    incident_flux_W_m2_value,
    radius_micron,
    Tmin_K=2.0,
    Tmax_K=10000.0,
):
    """
    Independent direct root solve expressed in incident-flux space.

    Used only for validation of the lookup.
    """

    F = float(incident_flux_W_m2_value)

    if F < 0.0:
        raise ValueError("incident flux must be non-negative.")

    if F == 0.0:
        return 0.0

    q_source = float(
        thermal_solver.source_mean_qabs(radius_micron)
    )

    def residual(T):
        emitted = (
            4.0
            * np.pi
            * thermal_solver.emission_integral(
                temperature_K=T,
                radius_micron=radius_micron,
            )
        )

        absorbed = F * q_source

        return absorbed - emitted

    f_lo = residual(Tmin_K)
    f_hi = residual(Tmax_K)

    if f_lo == 0.0:
        return float(Tmin_K)

    if f_hi == 0.0:
        return float(Tmax_K)

    if np.sign(f_lo) == np.sign(f_hi):
        return np.nan

    return float(
        brentq(
            residual,
            Tmin_K,
            Tmax_K,
        )
    )