"""
V4.1 dust-composition mixture utilities.

The mixture is formed at the physical response level, before any photometric
nuisance fitting. A single graphite fraction is shared by W1 and W2:

    R_band = f_graphite * R_band_graphite
           + (1-f_graphite) * R_band_silicate

Each pure component must already include its own Q_abs, radiative-equilibrium
temperature solution, grain-size integration, irreversible sublimation, and
3-D echo transfer function.

IMPORTANT:
f_graphite is an effective relative population normalization in the current
relative-response V4 framework. It is NOT yet a dust mass fraction or covering
factor.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MixedBandResponse:
    time_obs_days: np.ndarray
    response: np.ndarray
    f_graphite: float


def mix_response_arrays(
    graphite_response,
    silicate_response,
    f_graphite: float,
):
    """Mix two already-physical response arrays with one shared fraction."""
    f = float(f_graphite)
    if not np.isfinite(f) or not (0.0 <= f <= 1.0):
        raise ValueError("f_graphite must be finite and within [0, 1].")

    g = np.asarray(graphite_response, dtype=float)
    s = np.asarray(silicate_response, dtype=float)
    if g.shape != s.shape:
        raise ValueError(
            f"Graphite/silicate response shapes differ: {g.shape} vs {s.shape}"
        )
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(s)):
        raise ValueError("Responses contain non-finite values.")

    return f * g + (1.0 - f) * s


def mix_echoes(graphite_echo, silicate_echo, f_graphite: float):
    """
    Mix two echo objects that share exactly the same observer-time grid.

    Only the time_obs_days and response attributes are required, so this stays
    independent of the concrete echo dataclass used by V4.
    """
    tg = np.asarray(graphite_echo.time_obs_days, dtype=float)
    ts = np.asarray(silicate_echo.time_obs_days, dtype=float)

    if tg.shape != ts.shape or not np.allclose(tg, ts, rtol=0.0, atol=1e-10):
        raise ValueError("Graphite and silicate echoes must share one time grid.")

    response = mix_response_arrays(
        graphite_echo.response, silicate_echo.response, f_graphite
    )
    return MixedBandResponse(
        time_obs_days=tg.copy(),
        response=response,
        f_graphite=float(f_graphite),
    )


def mix_wise_echoes(
    graphite_w1,
    graphite_w2,
    silicate_w1,
    silicate_w2,
    f_graphite: float,
):
    """
    Apply the SAME f_graphite to W1 and W2.

    This is deliberate: no band-dependent mixture coefficient is permitted.
    """
    w1 = mix_echoes(graphite_w1, silicate_w1, f_graphite)
    w2 = mix_echoes(graphite_w2, silicate_w2, f_graphite)

    if not np.allclose(
        w1.time_obs_days, w2.time_obs_days, rtol=0.0, atol=1e-10
    ):
        raise ValueError("W1 and W2 mixed responses must share one time grid.")

    return w1, w2
