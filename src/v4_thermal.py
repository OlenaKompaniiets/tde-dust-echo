"""
v4_thermal.py
=============

Radiative-equilibrium dust-temperature solver for the physical V4 model.

This module contains NO WISE fitting and NO phenomenological dust
temperatures.

For a grain of radius a at distance r from a central source:

    F_bol = L_bol / (4 pi r^2)

The illuminating spectral shape S_lambda is normalized such that

    integral S_lambda d(lambda) = 1.

Define

    <Qabs>_src = integral Qabs(lambda,a) S_lambda d(lambda).

Radiative equilibrium is then

    F_bol <Qabs>_src
        =
    4 pi integral Qabs(lambda,a) B_lambda(Tdust) d(lambda).

Grain cross-sectional factors cancel.

All wavelength integrations are performed internally in SI units.
"""

import numpy as np
from scipy.optimize import brentq


# ============================================================
# Physical constants -- SI
# ============================================================

H = 6.62607015e-34          # J s
C = 2.99792458e8            # m s^-1
K_B = 1.380649e-23          # J K^-1

PC_M = 3.085677581491367e16
ERG_S_TO_W = 1.0e-7
MICRON_M = 1.0e-6


# ============================================================
# Planck function
# ============================================================

def planck_lambda_si(wavelength_m, temperature_K):
    """
    Planck spectral radiance B_lambda.

    Parameters
    ----------
    wavelength_m : array-like
        Wavelength in metres.

    temperature_K : float
        Temperature in K.

    Returns
    -------
    B_lambda : ndarray
        W m^-3 sr^-1
    """

    lam = np.asarray(
        wavelength_m,
        dtype=float,
    )

    T = float(temperature_K)

    if np.any(lam <= 0):
        raise ValueError(
            "Wavelength must be positive."
        )

    if T <= 0:
        raise ValueError(
            "Temperature must be positive."
        )

    x = H * C / (
        lam * K_B * T
    )

    # Avoid numerical overflow in exp().
    x_clip = np.minimum(
        x,
        700.0,
    )

    denominator = np.expm1(
        x_clip
    )

    B = (
        2.0 * H * C**2
        / lam**5
        / denominator
    )

    # For x > 700 the physical value is effectively zero.
    B = np.where(
        x > 700.0,
        0.0,
        B,
    )

    return B


# ============================================================
# Numerical integration helper
# ============================================================

def integrate_log_wavelength(
    wavelength_m,
    y,
):
    """
    Integrate y(lambda) d(lambda).

    Input wavelength grid may be logarithmic or irregular,
    but must be strictly increasing.
    """

    lam = np.asarray(
        wavelength_m,
        dtype=float,
    )

    yy = np.asarray(
        y,
        dtype=float,
    )

    if lam.ndim != 1:
        raise ValueError(
            "wavelength_m must be 1D."
        )

    if yy.shape != lam.shape:
        raise ValueError(
            "y must have same shape as wavelength_m."
        )

    if np.any(np.diff(lam) <= 0):
        raise ValueError(
            "Wavelength grid must be increasing."
        )

    return np.trapezoid(
        yy,
        lam,
    )


# ============================================================
# Illuminating blackbody SED
# ============================================================

def normalized_blackbody_sed(
    wavelength_m,
    temperature_K,
):
    """
    Return normalized blackbody spectral shape S_lambda:

        integral S_lambda d(lambda) = 1.

    This describes spectral SHAPE only.

    The absolute source luminosity is supplied independently
    as L_bol.
    """

    lam = np.asarray(
        wavelength_m,
        dtype=float,
    )

    B = planck_lambda_si(
        lam,
        temperature_K,
    )

    norm = integrate_log_wavelength(
        lam,
        B,
    )

    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError(
            "Could not normalize source SED."
        )

    return B / norm


# ============================================================
# Thermal engine
# ============================================================

class DustThermalEquilibrium:
    """
    Radiative-equilibrium solver using a DustOpacity object
    from v4_opacity.py.
    """

    def __init__(
        self,
        opacity,
        source_temperature_K=30000.0,
        lambda_min_micron=0.001,
        lambda_max_micron=1000.0,
        n_lambda=1201,
    ):

        self.opacity = opacity

        self.source_temperature_K = float(
            source_temperature_K
        )

        if self.source_temperature_K <= 0:
            raise ValueError(
                "source_temperature_K must be positive."
            )

        # Never evaluate outside opacity table.
        lo = max(
            float(lambda_min_micron),
            opacity.lambda_min_micron,
        )

        hi = min(
            float(lambda_max_micron),
            opacity.lambda_max_micron,
        )

        if hi <= lo:
            raise ValueError(
                "Invalid wavelength range."
            )

        self.wavelength_micron = np.geomspace(
            lo,
            hi,
            int(n_lambda),
        )

        self.wavelength_m = (
            self.wavelength_micron
            * MICRON_M
        )

        self.source_sed = normalized_blackbody_sed(
            self.wavelength_m,
            self.source_temperature_K,
        )

    # --------------------------------------------------------
    # Source absorption
    # --------------------------------------------------------

    def source_mean_qabs(
        self,
        radius_micron,
    ):
        """
        Planck-weighted absorption efficiency for the
        illuminating source spectrum.
        """

        a = np.asarray(
            radius_micron,
            dtype=float,
        )

        q = self.opacity.q_abs(
            self.wavelength_micron[:, None],
            np.atleast_1d(a)[None, :],
        )

        mean_q = np.trapezoid(
            q * self.source_sed[:, None],
            self.wavelength_m,
            axis=0,
        )

        if np.ndim(radius_micron) == 0:
            return float(mean_q[0])

        return mean_q

    # --------------------------------------------------------
    # Thermal emission integral
    # --------------------------------------------------------

    def emission_integral(
        self,
        temperature_K,
        radius_micron,
    ):
        """
        Return

            integral Qabs(lambda,a) B_lambda(T) d(lambda)

        in W m^-2 sr^-1.
        """

        q = self.opacity.q_abs(
            self.wavelength_micron,
            float(radius_micron),
        )

        B = planck_lambda_si(
            self.wavelength_m,
            temperature_K,
        )

        return np.trapezoid(
            q * B,
            self.wavelength_m,
        )

    # --------------------------------------------------------
    # Equilibrium residual
    # --------------------------------------------------------

    def equilibrium_residual(
        self,
        temperature_K,
        luminosity_erg_s,
        distance_pc,
        radius_micron,
    ):
        """
        Residual:

            absorbed flux - emitted flux

        Equilibrium occurs at residual = 0.
        """

        L_W = (
            float(luminosity_erg_s)
            * ERG_S_TO_W
        )

        r_m = (
            float(distance_pc)
            * PC_M
        )

        if L_W < 0:
            raise ValueError(
                "Luminosity cannot be negative."
            )

        if r_m <= 0:
            raise ValueError(
                "distance_pc must be positive."
            )

        mean_q = self.source_mean_qabs(
            radius_micron
        )

        absorbed = (
            L_W
            / (4.0 * np.pi * r_m**2)
            * mean_q
        )

        emitted = (
            4.0
            * np.pi
            * self.emission_integral(
                temperature_K,
                radius_micron,
            )
        )

        return absorbed - emitted

    # --------------------------------------------------------
    # Solve Tdust
    # --------------------------------------------------------

    def equilibrium_temperature(
        self,
        luminosity_erg_s,
        distance_pc,
        radius_micron,
        Tmin_K=2.0,
        Tmax_K=10000.0,
    ):
        """
        Solve radiative equilibrium for one grain size.

        No sublimation is applied here.

        Sublimation/destruction belongs to a separate physical
        layer and will be implemented only after this solver
        passes its sanity tests.
        """

        L = float(
            luminosity_erg_s
        )

        if L == 0:
            return 0.0

        f_lo = self.equilibrium_residual(
            Tmin_K,
            L,
            distance_pc,
            radius_micron,
        )

        f_hi = self.equilibrium_residual(
            Tmax_K,
            L,
            distance_pc,
            radius_micron,
        )

        if f_lo < 0:
            # Equilibrium below Tmin.
            return float(Tmin_K)

        if f_hi > 0:
            raise RuntimeError(
                "Equilibrium temperature exceeds Tmax_K. "
                "This may indicate a sublimating grain or an "
                "insufficient temperature bracket."
            )

        T = brentq(
            lambda temp:
                self.equilibrium_residual(
                    temp,
                    L,
                    distance_pc,
                    radius_micron,
                ),
            Tmin_K,
            Tmax_K,
            xtol=1.0e-7,
            rtol=1.0e-10,
            maxiter=200,
        )

        return float(T)

    # --------------------------------------------------------
    # Vector over grain sizes
    # --------------------------------------------------------

    def temperature_for_sizes(
        self,
        luminosity_erg_s,
        distance_pc,
        radius_micron,
        Tmin_K=2.0,
        Tmax_K=10000.0,
    ):
        """
        Solve equilibrium temperature independently for
        every grain size.
        """

        a = np.asarray(
            radius_micron,
            dtype=float,
        )

        result = np.empty_like(
            a,
            dtype=float,
        )

        for i, aa in np.ndenumerate(a):

            result[i] = self.equilibrium_temperature(
                luminosity_erg_s=luminosity_erg_s,
                distance_pc=distance_pc,
                radius_micron=float(aa),
                Tmin_K=Tmin_K,
                Tmax_K=Tmax_K,
            )

        return result