"""
Observer-frame light-travel-time mapping for the
UGC 11487 physical dust-echo V4 model.

TIME CONVENTION
---------------
The thermal/dust-response cube is indexed by dust-local rest-frame time:

    t_dust = t_source + r/c

because the heating luminosity is evaluated as

    L_heat(t_dust, r) = L_source(t_dust - r/c).

For a dust element with observer-frame Cartesian coordinate z_obs
(observer located toward +z), the arrival time relative to a photon
emitted directly from the origin is

    t_arrival_rest = t_dust - z_obs/c

                   = t_source + (r - z_obs)/c.

Therefore the full echo delay (r - z_obs)/c must NOT be added to
t_dust; doing so would count r/c twice.

Observed-frame relative times are dilated by

    t_arrival_obs = (1 + z_cosmo) * t_arrival_rest.

This module performs only the timing/remapping operation.
It does not assign an absolute dust normalization.
"""

from dataclasses import dataclass

import numpy as np


PC_TO_LIGHT_DAYS = 3.085677581491367e16 / 299792458.0 / 86400.0


@dataclass
class ObserverEchoResult:
    time_obs_days: np.ndarray
    response: np.ndarray
    total_input_weight: float
    total_output_weight: float


def dust_to_observer_delay_rest_days(
    z_obs_pc,
):
    """
    Propagation term from dust-local time to observer arrival time.

    Since t_dust already contains the source-to-dust r/c travel time,

        t_arrival_rest = t_dust - z_obs/c.

    Positive z_obs is the near side of the dust distribution.
    """

    z_obs_pc = np.asarray(
        z_obs_pc,
        dtype=float,
    )

    return (
        -z_obs_pc
        * PC_TO_LIGHT_DAYS
    )


def full_echo_delay_rest_days(
    r_pc,
    z_obs_pc,
):
    """
    Full source-to-dust-to-observer echo delay relative to direct light:

        tau_echo = (r - z_obs) / c.

    This is useful for diagnostics and tests.

    IMPORTANT:
    Do not add this full delay to a response already indexed by
    dust-local time.
    """

    r_pc = np.asarray(
        r_pc,
        dtype=float,
    )

    z_obs_pc = np.asarray(
        z_obs_pc,
        dtype=float,
    )

    r_pc, z_obs_pc = np.broadcast_arrays(
        r_pc,
        z_obs_pc,
    )

    return (
        (r_pc - z_obs_pc)
        * PC_TO_LIGHT_DAYS
    )


def dust_time_to_observer_time_days(
    t_dust_rest_days,
    z_obs_pc,
    redshift=0.0,
):
    """
    Convert dust-local rest-frame times to observed relative times.

        t_obs = (1 + z) * (t_dust - z_obs/c)
    """

    if redshift < 0.0:
        raise ValueError(
            "redshift must be non-negative."
        )

    t_dust = np.asarray(
        t_dust_rest_days,
        dtype=float,
    )

    delay = dust_to_observer_delay_rest_days(
        z_obs_pc
    )

    return (
        (1.0 + float(redshift))
        * (
            t_dust[..., None]
            + delay
        )
    )


def deposit_linear(
    output_time_days,
    arrival_time_days,
    values,
):
    """
    Deposit weighted samples onto a regular output time grid using
    linear interpolation between the two nearest bins.

    Integrated discrete weight is conserved except for samples whose
    arrival times lie outside the supplied output grid.
    """

    output_time = np.asarray(
        output_time_days,
        dtype=float,
    )

    arrival_time = np.asarray(
        arrival_time_days,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    if output_time.ndim != 1:
        raise ValueError(
            "output_time_days must be 1D."
        )

    if output_time.size < 2:
        raise ValueError(
            "At least two output time samples are required."
        )

    dt = np.diff(output_time)

    if not np.all(dt > 0.0):
        raise ValueError(
            "output_time_days must be strictly increasing."
        )

    if not np.allclose(
        dt,
        dt[0],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "output_time_days must be uniformly spaced."
        )

    if arrival_time.shape != values.shape:
        raise ValueError(
            "arrival_time_days and values must have identical shapes."
        )

    result = np.zeros_like(
        output_time,
        dtype=float,
    )

    finite = (
        np.isfinite(arrival_time)
        & np.isfinite(values)
        & (values != 0.0)
    )

    if not np.any(finite):
        return result

    t = arrival_time[finite]
    v = values[finite]

    x = (
        (t - output_time[0])
        / dt[0]
    )

    left = np.floor(x).astype(int)
    frac = x - left

    # Interior points: split linearly between adjacent bins.
    interior = (
        (left >= 0)
        & (left < output_time.size - 1)
        & ~np.isclose(t, output_time[-1], rtol=0., atol=max(1e-10, abs(dt[0])*1e-10))
    )

    if np.any(interior):
        li = left[interior]
        fi = frac[interior]
        vi = v[interior]

        np.add.at(
            result,
            li,
            vi * (1.0 - fi),
        )

        np.add.at(
            result,
            li + 1,
            vi * fi,
        )

    # Exact final-grid-point arrivals.
    final = (
        np.isclose(
            t,
            output_time[-1],
            rtol=0.0,
            atol=max(
                1.0e-10,
                abs(dt[0]) * 1.0e-10,
            ),
        )
    )

    if np.any(final):
        result[-1] += float(np.sum(v[final]))

    return result


def map_clump_responses_to_observer(
    dust_time_rest_days,
    clump_response,
    z_obs_pc,
    clump_weight=None,
    redshift=0.0,
    output_time_obs_days=None,
):
    """
    Map time-dependent emission from many clumps to observer time.

    Parameters
    ----------
    dust_time_rest_days : array, shape (Nt,)
        Dust-local rest-frame time grid.

    clump_response : array, shape (Nt, Nc)
        Emission/response of each clump as a function of dust-local time.

    z_obs_pc : array, shape (Nc,)
        Observer-frame z coordinate of each clump.
        Positive z is the near side.

    clump_weight : optional array, shape (Nc,)
        Relative clump weights. If omitted, all clumps have equal weight.

    redshift : float
        Cosmological redshift.

    output_time_obs_days : optional array, shape (Nout,)
        Uniform observed-frame output grid. If omitted, a grid with
        approximately the input rest-frame cadence times (1+z) is built
        to contain all arrival times.

    Returns
    -------
    ObserverEchoResult
    """

    if redshift < 0.0:
        raise ValueError(
            "redshift must be non-negative."
        )

    t_dust = np.asarray(
        dust_time_rest_days,
        dtype=float,
    )

    response = np.asarray(
        clump_response,
        dtype=float,
    )

    z_obs = np.asarray(
        z_obs_pc,
        dtype=float,
    )

    if t_dust.ndim != 1:
        raise ValueError(
            "dust_time_rest_days must be 1D."
        )

    if response.ndim != 2:
        raise ValueError(
            "clump_response must have shape (Nt, Nc)."
        )

    if response.shape[0] != t_dust.size:
        raise ValueError(
            "First clump_response dimension must match time grid."
        )

    if response.shape[1] != z_obs.size:
        raise ValueError(
            "Second clump_response dimension must match z_obs_pc."
        )

    if clump_weight is None:
        weight = np.ones(
            z_obs.size,
            dtype=float,
        )
    else:
        weight = np.asarray(
            clump_weight,
            dtype=float,
        )

        if weight.shape != z_obs.shape:
            raise ValueError(
                "clump_weight must match z_obs_pc."
            )

    if np.any(~np.isfinite(weight)):
        raise ValueError(
            "clump_weight contains non-finite values."
        )

    if np.any(weight < 0.0):
        raise ValueError(
            "clump_weight must be non-negative."
        )

    arrival = dust_time_to_observer_time_days(
        t_dust,
        z_obs,
        redshift=redshift,
    )

    weighted_response = (
        response
        * weight[None, :]
    )

    if output_time_obs_days is None:

        if t_dust.size < 2:
            raise ValueError(
                "At least two time samples are required "
                "to construct an output grid."
            )

        dt_rest = np.median(
            np.diff(t_dust)
        )

        dt_obs = (
            dt_rest
            * (1.0 + float(redshift))
        )

        tmin = np.floor(
            np.min(arrival) / dt_obs
        ) * dt_obs

        tmax = np.ceil(
            np.max(arrival) / dt_obs
        ) * dt_obs

        nout = (
            int(
                np.round(
                    (tmax - tmin) / dt_obs
                )
            )
            + 1
        )

        output_time = (
            tmin
            + np.arange(
                nout,
                dtype=float,
            )
            * dt_obs
        )

    else:
        output_time = np.asarray(
            output_time_obs_days,
            dtype=float,
        )

    if not np.allclose(np.diff(t_dust), np.diff(t_dust)[0], rtol=1e-9, atol=1e-10):
        raise ValueError("Uniform dust cadence required")
    if not np.allclose(np.diff(output_time), np.diff(t_dust)[0]*(1+redshift), rtol=1e-9, atol=1e-10):
        raise ValueError("Output cadence must equal dust cadence times (1+z)")
    observed = deposit_linear(
        output_time_days=output_time,
        arrival_time_days=arrival.ravel(),
        values=weighted_response.ravel(),
    )

    return ObserverEchoResult(
        time_obs_days=output_time,
        response=observed,
        total_input_weight=float(
            np.sum(weighted_response)
        ),
        total_output_weight=float(
            np.sum(observed)
        ),
    )