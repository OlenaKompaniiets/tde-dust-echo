"""
Time-dependent physical dust response for UGC 11487 V4.

Physics:
    source luminosity
        -> source-to-dust light-travel delay
        -> incident bolometric flux
        -> radiative-equilibrium temperature from validated lookup
        -> irreversible sublimation

No WISE data or empirical MIR timing constraints enter this module.
"""

from dataclasses import dataclass

import numpy as np

from src.v4_sublimation import sublimation_temperature_K
from src.v4_thermal_lookup import (
    build_thermal_lookup,
    incident_flux_W_m2,
    lookup_temperature_from_flux,
)


C_M_S = 299792458.0
PC_M = 3.085677581491367e16
DAY_S = 86400.0

PC_TO_LIGHT_DAYS = PC_M / C_M_S / DAY_S


@dataclass
class DustResponseResult:
    time_days: np.ndarray
    radius_pc: np.ndarray
    grain_radius_micron: np.ndarray

    luminosity_source_erg_s: np.ndarray
    luminosity_heating_erg_s: np.ndarray

    temperature_K: np.ndarray
    alive: np.ndarray

    sublimation_temperature_K: float

    composition: str
    density_cm3: float


def retarded_luminosity(
    time_days,
    luminosity_erg_s,
    radius_pc,
):
    """
    Source luminosity seen by dust at each radius.

    L_heat(t,r) = L_source(t - r/c)

    This contains ONLY the source-to-dust propagation delay.
    Dust-to-observer propagation belongs to the later echo calculation.

    Returns
    -------
    ndarray, shape (Nt, Nr)
    """

    time = np.asarray(time_days, dtype=float)
    luminosity = np.asarray(luminosity_erg_s, dtype=float)
    radius = np.asarray(radius_pc, dtype=float)

    if time.ndim != 1 or luminosity.ndim != 1:
        raise ValueError("time_days and luminosity_erg_s must be 1D.")

    if time.size != luminosity.size:
        raise ValueError(
            "time_days and luminosity_erg_s must have the same length."
        )

    if radius.ndim != 1:
        raise ValueError("radius_pc must be 1D.")

    if np.any(radius <= 0.0):
        raise ValueError("radius_pc must be strictly positive.")

    if np.any(luminosity < 0.0):
        raise ValueError("luminosity_erg_s must be non-negative.")

    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time_days must be strictly increasing.")

    delays = radius * PC_TO_LIGHT_DAYS

    query_time = (
        time[:, None]
        - delays[None, :]
    )

    result = np.empty(
        (time.size, radius.size),
        dtype=float,
    )

    for j in range(radius.size):
        result[:, j] = np.interp(
            query_time[:, j],
            time,
            luminosity,
            left=0.0,
            right=0.0,
        )

    return result


def compute_temperature_cube(
    thermal_solver,
    luminosity_heating_erg_s,
    radius_pc,
    grain_radius_micron,
    composition,
    density_cm3,
    Tmax_K=10000.0,
    n_lookup_temperature=1600,
):
    """
    Fast radiative-equilibrium + irreversible-sublimation calculation.

    The equilibrium relation is tabulated once for each grain size using
    exactly the same thermal physics as DustThermalEquilibrium.

    A grain is destroyed when the equilibrium temperature required by
    the incident radiation reaches or exceeds T_sub.  Destruction is
    irreversible.

    Returns
    -------
    temperature_K : ndarray, shape (Nt, Nr, Na)

        Surviving grains:
            equilibrium temperature.

        Destruction timestep:
            T_sub, stored as the threshold diagnostic.

        After destruction:
            NaN, because the grain no longer exists.

        Zero illumination:
            0 K under the transient-heating convention used here.

    alive : ndarray of bool, shape (Nt, Nr, Na)
    """

    L_heat = np.asarray(
        luminosity_heating_erg_s,
        dtype=float,
    )

    radius = np.asarray(
        radius_pc,
        dtype=float,
    )

    grains = np.asarray(
        grain_radius_micron,
        dtype=float,
    )

    if L_heat.ndim != 2:
        raise ValueError(
            "luminosity_heating_erg_s must have shape (Nt, Nr)."
        )

    if radius.ndim != 1:
        raise ValueError("radius_pc must be 1D.")

    if grains.ndim != 1:
        raise ValueError("grain_radius_micron must be 1D.")

    if L_heat.shape[1] != radius.size:
        raise ValueError(
            "Second dimension of luminosity_heating_erg_s "
            "must match radius_pc."
        )

    if np.any(~np.isfinite(L_heat)):
        raise ValueError(
            "luminosity_heating_erg_s contains non-finite values."
        )

    if np.any(L_heat < 0.0):
        raise ValueError(
            "luminosity_heating_erg_s must be non-negative."
        )

    if np.any(radius <= 0.0):
        raise ValueError(
            "radius_pc must be strictly positive."
        )

    if np.any(grains <= 0.0):
        raise ValueError(
            "grain_radius_micron must be strictly positive."
        )

    T_sub = float(
        sublimation_temperature_K(
            density_cm3=density_cm3,
            composition=composition,
        )
    )

    # ------------------------------------------------------------
    # Build the validated equilibrium lookup once.
    # ------------------------------------------------------------

    lookup = build_thermal_lookup(
        thermal_solver=thermal_solver,
        grain_radius_micron=grains,
        Tmin_K=2.0,
        Tmax_K=Tmax_K,
        n_temperature=n_lookup_temperature,
    )

    if T_sub >= lookup.temperature_grid_K[-1]:
        raise ValueError("Lookup ceiling must exceed sublimation threshold")

    Nt, Nr = L_heat.shape
    Na = grains.size

    temperature = np.zeros(
        (Nt, Nr, Na),
        dtype=float,
    )

    alive = np.ones(
        (Nt, Nr, Na),
        dtype=bool,
    )

    # ------------------------------------------------------------
    # Incident flux cube has only (time, radius).
    # ------------------------------------------------------------

    flux = incident_flux_W_m2(
        luminosity_erg_s=L_heat,
        distance_pc=radius[None, :],
    )

    # ------------------------------------------------------------
    # Evaluate one grain size at a time.
    # This is vectorized over ALL times and radii.
    # ------------------------------------------------------------

    for k in range(Na):

        T_eq = lookup_temperature_from_flux(
            lookup=lookup,
            incident_flux=flux,
            grain_index=k,
        )

        # Only flux ABOVE the lookup ceiling implies overheating.
        # Below Tmin is explicitly cold/MIR-dark, never destroyed.
        high_flux_outside_lookup = (
            flux > 10.**lookup.log10_flux_grid_W_m2[k, -1]
        )

        destruction_condition = (
            high_flux_outside_lookup
            | (T_eq >= T_sub)
        )

        # Irreversible destruction along the time axis.
        destroyed_ever = np.maximum.accumulate(
            destruction_condition,
            axis=0,
        )

        alive[:, :, k] = ~destroyed_ever

        T_store = np.asarray(
            T_eq,
            dtype=float,
        ).copy()

        # Store T_sub at the FIRST destruction timestep.
        first_destruction = (
            destruction_condition
            & ~np.vstack(
                [
                    np.zeros(
                        (1, Nr),
                        dtype=bool,
                    ),
                    destroyed_ever[:-1, :],
                ]
            )
        )

        T_store[first_destruction] = T_sub

        # No grain exists after the destruction timestep.
        after_destruction = (
            destroyed_ever
            & ~first_destruction
        )

        T_store[after_destruction] = np.nan

        temperature[:, :, k] = T_store

    return temperature, alive


def build_dust_response(
    time_days,
    luminosity_erg_s,
    radius_pc,
    grain_radius_micron,
    thermal_solver,
    composition,
    density_cm3,
    Tmax_K=10000.0,
):
    """
    Build the radial time-dependent physical dust response.
    """

    time = np.asarray(
        time_days,
        dtype=float,
    )

    luminosity = np.asarray(
        luminosity_erg_s,
        dtype=float,
    )

    radius = np.asarray(
        radius_pc,
        dtype=float,
    )

    grains = np.asarray(
        grain_radius_micron,
        dtype=float,
    )

    L_heat = retarded_luminosity(
        time_days=time,
        luminosity_erg_s=luminosity,
        radius_pc=radius,
    )

    temperature, alive = compute_temperature_cube(
        thermal_solver=thermal_solver,
        luminosity_heating_erg_s=L_heat,
        radius_pc=radius,
        grain_radius_micron=grains,
        composition=composition,
        density_cm3=density_cm3,
        Tmax_K=Tmax_K,
    )

    T_sub = float(
        sublimation_temperature_K(
            density_cm3=density_cm3,
            composition=composition,
        )
    )

    return DustResponseResult(
        time_days=time,
        radius_pc=radius,
        grain_radius_micron=grains,
        luminosity_source_erg_s=luminosity,
        luminosity_heating_erg_s=L_heat,
        temperature_K=temperature,
        alive=alive,
        sublimation_temperature_K=T_sub,
        composition=composition,
        density_cm3=float(density_cm3),
    )


def minimum_surviving_radius_pc(
    response: DustResponseResult,
):
    """
    Minimum radius that survives through the complete simulated history,
    separately for every grain size.

    Returns
    -------
    ndarray, shape (Na,)

    NaN means that no sampled radius survives for that grain size.
    """

    radius = response.radius_pc
    final_alive = response.alive[-1, :, :]

    Na = response.grain_radius_micron.size

    result = np.full(
        Na,
        np.nan,
        dtype=float,
    )

    for k in range(Na):

        mask = final_alive[:, k]

        if np.any(mask):
            result[k] = np.min(
                radius[mask]
            )

    return result


def peak_temperature_by_radius_and_size(
    response: DustResponseResult,
):
    """
    Maximum finite temperature reached at every (radius, grain-size)
    location.

    The destruction timestep contains T_sub; later destroyed states are
    NaN and are ignored.

    Returns
    -------
    ndarray, shape (Nr, Na)
    """

    T = np.asarray(
        response.temperature_K,
        dtype=float,
    )

    with np.errstate(all="ignore"):
        peak = np.nanmax(
            T,
            axis=0,
        )

    return peak