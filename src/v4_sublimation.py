"""
V4 dust sublimation / survival physics.

This module deliberately separates:

    radiative equilibrium:
        L(t), r, a, composition -> T_eq

from:

    material survival:
        T_eq, gas density, composition -> alive / destroyed

No WISE observables enter this module.
No fitted dust temperatures enter this module.
No phenomenological dust zones enter this module.

The default sublimation-temperature prescription follows
Baskin & Laor (2018, MNRAS, 474, 1970; doi:10.1093/mnras/stx2850),
their equations (32) and (33), derived for solar abundances and
T_gas = 10^4 K.

IMPORTANT
---------
The ambient gas density is a physical hypothesis, not inferred here
from the WISE light curves.

Dust destruction is treated as irreversible over the modeled flare:
once a grain is destroyed, it remains destroyed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------
# Supported compositions
# ---------------------------------------------------------------------

SUPPORTED_COMPOSITIONS = ("graphite", "silicate")

# Validity range quoted for the simple analytic approximations
# in Baskin & Laor (2018).
BL18_N_MIN_CM3 = 1.0
BL18_N_MAX_CM3 = 1.0e12


# ---------------------------------------------------------------------
# Sublimation temperature
# ---------------------------------------------------------------------

def sublimation_temperature_K(
    density_cm3,
    composition: str,
):
    """
    Sublimation temperature from Baskin & Laor (2018).

    Parameters
    ----------
    density_cm3 : float or array-like
        Total gas density n_H [cm^-3].

    composition : {"graphite", "silicate"}
        Grain composition.

    Returns
    -------
    T_sub_K : float or ndarray
        Sublimation temperature [K].

    Notes
    -----
    Approximate analytic relations:

    graphite:
        T_sub = 81200 / (66.003 - ln n)

    silicate:
        T_sub = 68100 / (67.957 - ln n)

    where n is in cm^-3.

    These approximations assume solar abundances and T_gas = 10^4 K.
    Baskin & Laor (2018) quote accuracy better than 0.5% over
    1 < n < 10^12 cm^-3.
    """

    composition = composition.lower().strip()

    if composition not in SUPPORTED_COMPOSITIONS:
        raise ValueError(
            f"Unsupported composition '{composition}'. "
            f"Supported: {SUPPORTED_COMPOSITIONS}"
        )

    n = np.asarray(density_cm3, dtype=float)

    if np.any(~np.isfinite(n)):
        raise ValueError("density_cm3 contains non-finite values.")

    if np.any(n <= 0.0):
        raise ValueError("density_cm3 must be strictly positive.")

    if np.any((n < BL18_N_MIN_CM3) | (n > BL18_N_MAX_CM3)):
        raise ValueError(
            "density_cm3 lies outside the adopted validity range "
            f"[{BL18_N_MIN_CM3:.1e}, {BL18_N_MAX_CM3:.1e}] cm^-3."
        )

    ln_n = np.log(n)

    if composition == "graphite":
        T_sub = 81200.0 / (66.003 - ln_n)
    else:
        T_sub = 68100.0 / (67.957 - ln_n)

    if np.ndim(T_sub) == 0:
        return float(T_sub)

    return T_sub


# ---------------------------------------------------------------------
# Instantaneous threshold test
# ---------------------------------------------------------------------

def survives_temperature(
    temperature_K,
    density_cm3,
    composition: str,
):
    """
    Test whether dust is below the adopted sublimation threshold.

    Equality with T_sub is treated as destruction.

    Returns
    -------
    bool or ndarray of bool
        True  -> survives
        False -> at/above sublimation threshold
    """

    T = np.asarray(temperature_K, dtype=float)

    if np.any(~np.isfinite(T)):
        raise ValueError("temperature_K contains non-finite values.")

    if np.any(T < 0.0):
        raise ValueError("temperature_K must be non-negative.")

    T_sub = sublimation_temperature_K(
        density_cm3=density_cm3,
        composition=composition,
    )

    result = T < T_sub

    if np.ndim(result) == 0:
        return bool(result)

    return result


# ---------------------------------------------------------------------
# Irreversible survival history
# ---------------------------------------------------------------------

def irreversible_survival_history(
    temperature_K,
    density_cm3,
    composition: str,
    time_axis: int = 0,
):
    """
    Construct an irreversible grain-survival history.

    Once T_eq >= T_sub at any time, the grain is marked destroyed
    at that time and at every subsequent time.

    Parameters
    ----------
    temperature_K : ndarray
        Temperature history. The time dimension is specified by
        `time_axis`.

    density_cm3 : float or ndarray
        Ambient gas density [cm^-3]. It must be broadcastable to
        temperature_K.

    composition : {"graphite", "silicate"}

    time_axis : int
        Axis corresponding to increasing physical time.

    Returns
    -------
    alive : ndarray of bool
        True while the grain survives, False from the first
        threshold crossing onward.
    """

    T = np.asarray(temperature_K, dtype=float)

    if T.ndim == 0:
        raise ValueError(
            "temperature_K must contain a time dimension."
        )

    if np.any(~np.isfinite(T)):
        raise ValueError("temperature_K contains non-finite values.")

    if np.any(T < 0.0):
        raise ValueError("temperature_K must be non-negative.")

    axis = np.core.numeric.normalize_axis_index(time_axis, T.ndim)

    instantaneous_alive = survives_temperature(
        temperature_K=T,
        density_cm3=density_cm3,
        composition=composition,
    )

    # Move time to axis 0 so cumulative survival is unambiguous.
    inst = np.moveaxis(instantaneous_alive, axis, 0)

    # Logical cumulative AND:
    # once False occurs, all later entries remain False.
    alive = np.logical_and.accumulate(inst, axis=0)

    return np.moveaxis(alive, 0, axis)


# ---------------------------------------------------------------------
# Destruction-time diagnostics
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DestructionResult:
    destroyed: bool
    first_destroyed_index: int | None
    first_destroyed_time_days: float | None
    peak_temperature_K: float
    sublimation_temperature_K: float


def destruction_diagnostic(
    time_days,
    temperature_K,
    density_cm3,
    composition: str,
) -> DestructionResult:
    """
    Summarize destruction for one grain temperature history.

    Parameters
    ----------
    time_days : 1D array
        Increasing time grid [days].

    temperature_K : 1D array
        Grain equilibrium temperature on the same grid [K].

    density_cm3 : float
        Ambient gas density [cm^-3].

    composition : str
        Grain composition.

    Returns
    -------
    DestructionResult
    """

    time = np.asarray(time_days, dtype=float)
    temp = np.asarray(temperature_K, dtype=float)

    if time.ndim != 1 or temp.ndim != 1:
        raise ValueError(
            "time_days and temperature_K must both be 1D."
        )

    if time.size != temp.size:
        raise ValueError(
            "time_days and temperature_K must have equal length."
        )

    if time.size == 0:
        raise ValueError("Input arrays must not be empty.")

    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(temp)):
        raise ValueError("Inputs contain non-finite values.")

    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time_days must be strictly increasing.")

    T_sub = sublimation_temperature_K(
        density_cm3=density_cm3,
        composition=composition,
    )

    crossed = temp >= T_sub
    indices = np.flatnonzero(crossed)

    if indices.size == 0:
        return DestructionResult(
            destroyed=False,
            first_destroyed_index=None,
            first_destroyed_time_days=None,
            peak_temperature_K=float(np.max(temp)),
            sublimation_temperature_K=float(T_sub),
        )

    idx = int(indices[0])

    return DestructionResult(
        destroyed=True,
        first_destroyed_index=idx,
        first_destroyed_time_days=float(time[idx]),
        peak_temperature_K=float(np.max(temp)),
        sublimation_temperature_K=float(T_sub),
    )


# ---------------------------------------------------------------------
# Convenience density grid for sensitivity tests
# ---------------------------------------------------------------------

def standard_density_hypotheses_cm3():
    """
    Return the initial V4 density hypotheses.

    These are sensitivity-test hypotheses, not fitted values.
    """
    return np.array(
        [
            1.0e5,
            1.0e7,
            1.0e9,
            1.0e11,
        ],
        dtype=float,
    )