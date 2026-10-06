"""
WISE W1/W2 emission layer for the UGC 11487 physical dust-echo V4 model.

This module converts physical grain temperatures into WISE band responses
using the tabulated Equal-Energy RSR curves.  When a CALSPEC Vega reference
spectrum is supplied, W1 and W2 are placed on one deterministic WISE
Vega-system catalog-equivalent F_nu scale.

No empirical WISE light-curve quantities are used here.
No fitted dust temperatures are used here.
No arbitrary W1/W2 amplitude ratio is introduced.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from src.v4_thermal import planck_lambda_si


MICRON_M = 1.0e-6

# Official WISE Vega-system zero-magnitude flux densities [Jy].
WISE_FNU0_JY = {
    "W1": 309.54,
    "W2": 171.79,
}


@dataclass
class WiseBandpass:
    name: str
    wavelength_micron: np.ndarray
    response: np.ndarray


def load_wise_equal_energy_rsr(path, name):
    """
    Load an official WISE Equal-Energy relative spectral response file.

    Expected columns:
        wavelength [micron]
        RSR
        uncertainty [parts per thousand]

    The uncertainty column is not needed for deterministic forward
    convolution and is therefore not returned.
    """

    path = Path(path)

    data = np.loadtxt(
        path,
        comments="#",
    )

    wavelength = np.asarray(
        data[:, 0],
        dtype=float,
    )

    response = np.asarray(
        data[:, 1],
        dtype=float,
    )

    good = (
        np.isfinite(wavelength)
        & np.isfinite(response)
        & (wavelength > 0.0)
        & (response >= 0.0)
    )

    wavelength = wavelength[good]
    response = response[good]

    order = np.argsort(wavelength)

    wavelength = wavelength[order]
    response = response[order]

    if wavelength.size < 2:
        raise RuntimeError(
            f"Invalid WISE response file: {path}"
        )

    if np.max(response) <= 0.0:
        raise RuntimeError(
            f"WISE response is zero everywhere: {path}"
        )

    return WiseBandpass(
        name=str(name),
        wavelength_micron=wavelength,
        response=response,
    )


def load_wise_w1_w2(reference_dir):
    """
    Load the W1 and W2 Equal-Energy RSR curves.
    """

    reference_dir = Path(reference_dir)

    w1 = load_wise_equal_energy_rsr(
        reference_dir / "RSR-W1.EE.txt",
        "W1",
    )

    w2 = load_wise_equal_energy_rsr(
        reference_dir / "RSR-W2.EE.txt",
        "W2",
    )

    return w1, w2



def load_calspec_vega(path):
    """
    Load a CALSPEC Vega spectrum.

    CALSPEC FITS convention:
        WAVELENGTH : Angstrom
        FLUX       : erg s^-1 cm^-2 Angstrom^-1

    Only the spectral shape is required for the Vega-referenced
    natural-system operator, so no unit conversion of FLUX is needed.
    """
    path = Path(path)

    with fits.open(path) as hdul:
        table = hdul[1].data
        names = {name.upper(): name for name in table.names}

        wavelength_micron = np.asarray(
            table[names["WAVELENGTH"]],
            dtype=float,
        ) * 1.0e-4

        flux_lambda = np.asarray(
            table[names["FLUX"]],
            dtype=float,
        )

    good = (
        np.isfinite(wavelength_micron)
        & np.isfinite(flux_lambda)
        & (wavelength_micron > 0.0)
        & (flux_lambda > 0.0)
    )

    wavelength_micron = wavelength_micron[good]
    flux_lambda = flux_lambda[good]

    order = np.argsort(wavelength_micron)

    return (
        wavelength_micron[order],
        flux_lambda[order],
    )


def _loglog_interpolate_positive(x, xp, fp):
    """
    Positive log-log interpolation used for the smooth CALSPEC Vega SED.
    """
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if (
        np.min(x) < np.min(xp)
        or np.max(x) > np.max(xp)
    ):
        raise ValueError(
            "Reference spectrum does not cover the full bandpass."
        )

    return np.exp(
        np.interp(
            np.log(x),
            np.log(xp),
            np.log(fp),
        )
    )


def vega_band_signal(bandpass, vega_wavelength_micron, vega_flux_lambda):
    """
    Equal-Energy RSR signal of the CALSPEC Vega reference spectrum.

        S_Vega = integral F_lambda,Vega(lambda) R(lambda) d lambda
    """
    lam = np.asarray(
        bandpass.wavelength_micron,
        dtype=float,
    )

    rsr = np.asarray(
        bandpass.response,
        dtype=float,
    )

    vega = _loglog_interpolate_positive(
        lam,
        vega_wavelength_micron,
        vega_flux_lambda,
    )

    signal = np.trapezoid(
        vega * rsr,
        lam,
    )

    if signal <= 0.0:
        raise RuntimeError(
            f"Non-positive Vega band signal for {bandpass.name}."
        )

    return float(signal)


def wise_vega_scale_jy(
    bandpass,
    vega_wavelength_micron,
    vega_flux_lambda,
):
    """
    Multiplicative conversion from an Equal-Energy band signal expressed
    in the same arbitrary F_lambda normalization as the input spectrum
    to the WISE Vega-system catalog-equivalent flux-density scale.

        Fnu_cat[Jy] = Fnu0[Jy] * S_target / S_Vega

    IMPORTANT
    ---------
    For the present dust module, grain spectra still have deliberately
    relative absolute normalization. Therefore the returned Jy scale
    calibrates W1 versus W2 consistently but does NOT by itself turn
    the dust model into an absolute physical flux prediction.
    """
    name = str(bandpass.name)

    if name not in WISE_FNU0_JY:
        raise KeyError(
            f"No WISE zero point defined for band {name!r}."
        )

    s_vega = vega_band_signal(
        bandpass,
        vega_wavelength_micron,
        vega_flux_lambda,
    )

    return WISE_FNU0_JY[name] / s_vega

def grain_spectral_shape_lambda(
    opacity,
    wavelength_micron,
    grain_radius_micron,
    temperature_K,
):
    """
    Spectral emission shape of one spherical grain, apart from an
    overall geometric area factor common to wavelength.

        S_lambda ∝ a^2 Q_abs(lambda,a) B_lambda(T)

    The absolute normalization is deliberately not assigned here.
    This function is used for spectral/bandpass response.

    Returns SI spectral shape proportional to W m^-1 sr^-1 times a^2.
    """

    wavelength_micron = np.asarray(
        wavelength_micron,
        dtype=float,
    )

    if temperature_K <= 0.0:
        return np.zeros_like(
            wavelength_micron,
            dtype=float,
        )

    q_abs = opacity.q_abs(
        wavelength_micron=wavelength_micron,
        radius_micron=float(grain_radius_micron),
    )

    wavelength_m = (
        wavelength_micron
        * MICRON_M
    )

    B_lambda = planck_lambda_si(
        wavelength_m,
        float(temperature_K),
    )

    return (
        float(grain_radius_micron) ** 2
        * q_abs
        * B_lambda
    )


def band_response_for_grain(
    opacity,
    bandpass,
    grain_radius_micron,
    temperature_K,
    redshift=0.0,
    vega_wavelength_micron=None,
    vega_flux_lambda=None,
):
    """
    Relative WISE Equal-Energy band response for one grain.

    The bandpass is defined in observed wavelength.  Dust emission is
    evaluated at rest wavelength:

        lambda_rest = lambda_obs / (1 + z)

    The common cosmological amplitude factors are deliberately omitted
    because this layer currently validates spectral shape rather than
    absolute luminosity.

    Since the supplied RSR is explicitly Equal-Energy, the spectral
    energy distribution is integrated directly against the tabulated
    response.
    """

    if redshift < 0.0:
        raise ValueError(
            "redshift must be non-negative."
        )

    lambda_obs = bandpass.wavelength_micron

    lambda_rest = (
        lambda_obs
        / (1.0 + redshift)
    )

    spectral_shape = grain_spectral_shape_lambda(
        opacity=opacity,
        wavelength_micron=lambda_rest,
        grain_radius_micron=grain_radius_micron,
        temperature_K=temperature_K,
    )

    numerator = np.trapezoid(
        spectral_shape
        * bandpass.response,
        lambda_obs,
    )

    # Legacy/internal relative response if no Vega reference is supplied.
    # This preserves compatibility with existing shape-only tests.
    if (
        vega_wavelength_micron is None
        or vega_flux_lambda is None
    ):
        denominator = np.trapezoid(
            bandpass.response,
            lambda_obs,
        )

        if denominator <= 0.0:
            raise RuntimeError(
                f"Invalid integrated response for {bandpass.name}."
            )

        return float(
            numerator / denominator
        )

    # WISE Vega-system catalog-equivalent F_nu representation.
    scale_jy = wise_vega_scale_jy(
        bandpass,
        vega_wavelength_micron,
        vega_flux_lambda,
    )

    return float(
        numerator * scale_jy
    )


def grain_size_number_weights(
    grain_radius_micron,
    q=3.5,
):
    """
    Number weights for dn/da ∝ a^(-q) on an arbitrary positive size grid.

    Integration is performed using logarithmic cell boundaries.
    The returned weights sum to unity.

    Grain cross-sectional area is NOT included here because the a^2
    factor is already included in grain_spectral_shape_lambda().
    """

    a = np.asarray(
        grain_radius_micron,
        dtype=float,
    )

    if a.ndim != 1:
        raise ValueError(
            "grain_radius_micron must be 1D."
        )

    if a.size < 2:
        raise ValueError(
            "At least two grain sizes are required."
        )

    if np.any(a <= 0.0):
        raise ValueError(
            "grain sizes must be positive."
        )

    if np.any(np.diff(a) <= 0.0):
        raise ValueError(
            "grain sizes must be strictly increasing."
        )

    loga = np.log(a)

    edges_log = np.empty(
        a.size + 1,
        dtype=float,
    )

    edges_log[1:-1] = (
        0.5
        * (loga[:-1] + loga[1:])
    )

    # Requested nodes define hard physical support, not extrapolated cells.
    edges_log[0] = loga[0]
    edges_log[-1] = loga[-1]

    edges = np.exp(edges_log)

    if np.isclose(q, 1.0):
        weights = np.log(
            edges[1:] / edges[:-1]
        )
    else:
        exponent = 1.0 - q

        weights = (
            edges[1:] ** exponent
            - edges[:-1] ** exponent
        ) / exponent

    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights /= np.sum(weights)

    return weights


def population_band_response(
    opacity,
    bandpass,
    grain_radius_micron,
    temperature_K,
    alive=None,
    q=3.5,
    redshift=0.0,
    vega_wavelength_micron=None,
    vega_flux_lambda=None,
):
    """
    Band response of a grain-size distribution at one physical
    location/time.

    Each size has its own temperature.

    Parameters
    ----------
    temperature_K : array, shape (Na,)
    alive : optional bool array, shape (Na,)
    """

    grains = np.asarray(
        grain_radius_micron,
        dtype=float,
    )

    temperature = np.asarray(
        temperature_K,
        dtype=float,
    )

    if temperature.shape != grains.shape:
        raise ValueError(
            "temperature_K must match grain_radius_micron."
        )

    if alive is None:
        alive_mask = np.isfinite(temperature)
    else:
        alive_mask = np.asarray(
            alive,
            dtype=bool,
        )

        if alive_mask.shape != grains.shape:
            raise ValueError(
                "alive must match grain_radius_micron."
            )

        alive_mask = (
            alive_mask
            & np.isfinite(temperature)
        )

    weights = grain_size_number_weights(
        grains,
        q=q,
    )

    total = 0.0

    for k, a in enumerate(grains):

        if not alive_mask[k]:
            continue

        if temperature[k] <= 0.0:
            continue

        total += (
            weights[k]
            * band_response_for_grain(
                opacity=opacity,
                bandpass=bandpass,
                grain_radius_micron=a,
                temperature_K=temperature[k],
                redshift=redshift,
                vega_wavelength_micron=vega_wavelength_micron,
                vega_flux_lambda=vega_flux_lambda,
            )
        )

    return float(total)


def precompute_band_grid(
    opacity,
    bandpass,
    grain_radius_micron,
    temperature_grid_K,
    redshift=0.0,
    vega_wavelength_micron=None,
    vega_flux_lambda=None,
):
    """
    Precompute WISE band response as a function of
    grain size and temperature.

    Returns
    -------
    response_grid : ndarray, shape (Na, NT)
    """

    grains = np.asarray(
        grain_radius_micron,
        dtype=float,
    )

    temperatures = np.asarray(
        temperature_grid_K,
        dtype=float,
    )

    lambda_obs = np.asarray(
        bandpass.wavelength_micron,
        dtype=float,
    )

    rsr = np.asarray(
        bandpass.response,
        dtype=float,
    )

    lambda_rest = (
        lambda_obs
        / (1.0 + float(redshift))
    )

    lambda_rest_m = (
        lambda_rest
        * MICRON_M
    )

    if (
        vega_wavelength_micron is None
        or vega_flux_lambda is None
    ):
        denominator = np.trapezoid(
            rsr,
            lambda_obs,
        )

        if denominator <= 0.0:
            raise RuntimeError(
                f"Invalid integrated response for {bandpass.name}."
            )

        band_scale = 1.0 / denominator
    else:
        band_scale = wise_vega_scale_jy(
            bandpass,
            vega_wavelength_micron,
            vega_flux_lambda,
        )

    result = np.zeros(
        (
            grains.size,
            temperatures.size,
        ),
        dtype=float,
    )

    for ia, a in enumerate(grains):

        q_abs = np.asarray(
            opacity.q_abs(
                wavelength_micron=lambda_rest,
                radius_micron=float(a),
            ),
            dtype=float,
        )

        positive = (
            temperatures > 0.0
        )

        if not np.any(positive):
            continue

        T = temperatures[positive]

           # Shape: NT_positive x Nlambda.
        # planck_lambda_si() intentionally accepts one scalar
        # temperature at a time, so evaluate the temperature
        # lookup grid explicitly here.
        B = np.stack(
            [
                planck_lambda_si(
                    lambda_rest_m,
                    float(temp),
                )
                for temp in T
            ],
            axis=0,
        )

        spectra = (
            float(a) ** 2
            * q_abs[None, :]
            * B
        )

        result[
            ia,
            positive
        ] = (
            np.trapezoid(
                spectra
                * rsr[None, :],
                lambda_obs,
                axis=1,
            )
            * band_scale
        )

    return result


def fast_population_band_cube(
    opacity,
    bandpass,
    grain_radius_micron,
    temperature_K,
    alive,
    q=3.5,
    redshift=0.0,
    n_temperature=1200,
    vega_wavelength_micron=None,
    vega_flux_lambda=None,
):
    """
    Fast WISE response for an entire temperature cube.

    Parameters
    ----------
    temperature_K
        Array with shape (..., Na).

    alive
        Boolean array with the same shape.

    Returns
    -------
    band_response
        Array with shape temperature_K.shape[:-1].

    Notes
    -----
    The expensive wavelength convolution is performed only on
    a precomputed temperature grid.  The physical dust spectrum
    is unchanged; interpolation only accelerates evaluation.
    """

    grains = np.asarray(
        grain_radius_micron,
        dtype=float,
    )

    temperature = np.asarray(
        temperature_K,
        dtype=float,
    )

    alive = np.asarray(
        alive,
        dtype=bool,
    )

    if temperature.shape != alive.shape:
        raise ValueError(
            "temperature_K and alive must have identical shapes."
        )

    if temperature.shape[-1] != grains.size:
        raise ValueError(
            "Last temperature dimension must match grain-size grid."
        )

    valid = (
        alive
        & np.isfinite(temperature)
        & (temperature > 0.0)
    )

    output_shape = (
        temperature.shape[:-1]
    )

    if not np.any(valid):
        return np.zeros(
            output_shape,
            dtype=float,
        )

    positive_temperatures = (
        temperature[valid]
    )

    Tmin = max(
        2.0,
        float(np.min(positive_temperatures))
        * 0.95,
    )

    Tmax = (
        float(np.max(positive_temperatures))
        * 1.05
    )

    if Tmax <= Tmin:
        Tmax = Tmin + 1.0

    temperature_grid = np.geomspace(
        Tmin,
        Tmax,
        int(n_temperature),
    )

    response_grid = precompute_band_grid(
        opacity=opacity,
        bandpass=bandpass,
        grain_radius_micron=grains,
        temperature_grid_K=temperature_grid,
        redshift=redshift,
        vega_wavelength_micron=vega_wavelength_micron,
        vega_flux_lambda=vega_flux_lambda,
    )

    weights = grain_size_number_weights(
        grains,
        q=q,
    )

    output = np.zeros(
        output_shape,
        dtype=float,
    )

    log_grid = np.log(
        temperature_grid
    )

    for ia in range(
        grains.size
    ):

        mask = valid[..., ia]

        if not np.any(mask):
            continue

        T = temperature[..., ia]

        interpolated = np.zeros(
            output_shape,
            dtype=float,
        )

        interpolated[mask] = np.interp(
            np.log(T[mask]),
            log_grid,
            response_grid[ia],
        )

        output += (
            weights[ia]
            * interpolated
        )

    return output


# =====================================================================
# ABSOLUTE PHYSICAL EMISSION PATH
# =====================================================================
#
# These functions are deliberately separate from the historical
# shape-only response above.  They introduce no fitted normalization.
#
# For a spherical grain:
#
#   C_abs(lambda) = pi a^2 Q_abs(lambda,a)
#   L_lambda      = 4 pi C_abs B_lambda
#                 = 4 pi^2 a^2 Q_abs B_lambda
#
# where a is in metres and B_lambda is in W m^-3 sr^-1.
# Therefore L_lambda is returned in W m^-1.
#
# Observer-frame spectral flux density per unit wavelength:
#
#   F_lambda_obs(lambda_obs)
#       = L_lambda_rest(lambda_obs/(1+z))
#         / [4 pi D_L^2 (1+z)]
#
# with F_lambda_obs in W m^-3.
#
# No dust mass, grain count, covering factor, or clump normalization is
# assigned here.  Those belong to the population/geometry layer.


def grain_luminosity_lambda_si(
    opacity,
    wavelength_micron,
    grain_radius_micron,
    temperature_K,
):
    """
    Absolute rest-frame spectral luminosity of ONE spherical grain.

    Parameters
    ----------
    opacity
        Object exposing q_abs(wavelength_micron=..., radius_micron=...).
    wavelength_micron : array-like
        Rest-frame wavelength [micron].
    grain_radius_micron : float
        Physical grain radius [micron].
    temperature_K : float
        Grain temperature [K].

    Returns
    -------
    L_lambda : ndarray
        Spectral luminosity [W m^-1].

    Notes
    -----
    Uses Kirchhoff's law for a spherical grain:

        L_lambda = 4*pi*C_abs*B_lambda
                 = 4*pi^2*a^2*Q_abs*B_lambda,

    with C_abs = pi*a^2*Q_abs.
    """
    wavelength_micron = np.asarray(wavelength_micron, dtype=float)

    if np.any(wavelength_micron <= 0.0):
        raise ValueError("wavelength_micron must be positive.")

    if grain_radius_micron <= 0.0:
        raise ValueError("grain_radius_micron must be positive.")

    if temperature_K <= 0.0:
        return np.zeros_like(wavelength_micron, dtype=float)

    q_abs = np.asarray(
        opacity.q_abs(
            wavelength_micron=wavelength_micron,
            radius_micron=float(grain_radius_micron),
        ),
        dtype=float,
    )

    wavelength_m = wavelength_micron * MICRON_M
    a_m = float(grain_radius_micron) * MICRON_M

    B_lambda = planck_lambda_si(
        wavelength_m,
        float(temperature_K),
    )

    return (
        4.0
        * np.pi**2
        * a_m**2
        * q_abs
        * B_lambda
    )


def grain_flux_lambda_observer_si(
    opacity,
    wavelength_obs_micron,
    grain_radius_micron,
    temperature_K,
    luminosity_distance_m,
    redshift=0.0,
):
    """
    Absolute observer-frame F_lambda of ONE spherical grain.

    Parameters
    ----------
    wavelength_obs_micron : array-like
        Observed-frame wavelength [micron].
    luminosity_distance_m : float
        Luminosity distance [m].
    redshift : float
        Cosmological redshift.

    Returns
    -------
    F_lambda_obs : ndarray
        Observer-frame spectral flux density [W m^-3],
        i.e. W m^-2 per metre of wavelength.
    """
    wavelength_obs_micron = np.asarray(
        wavelength_obs_micron,
        dtype=float,
    )

    if np.any(wavelength_obs_micron <= 0.0):
        raise ValueError("wavelength_obs_micron must be positive.")

    if luminosity_distance_m <= 0.0:
        raise ValueError("luminosity_distance_m must be positive.")

    if redshift < 0.0:
        raise ValueError("redshift must be non-negative.")

    wavelength_rest_micron = (
        wavelength_obs_micron / (1.0 + float(redshift))
    )

    L_lambda_rest = grain_luminosity_lambda_si(
        opacity=opacity,
        wavelength_micron=wavelength_rest_micron,
        grain_radius_micron=grain_radius_micron,
        temperature_K=temperature_K,
    )

    return (
        L_lambda_rest
        / (
            4.0
            * np.pi
            * float(luminosity_distance_m)**2
            * (1.0 + float(redshift))
        )
    )


def grain_wise_flux_density_jy(
    opacity,
    bandpass,
    grain_radius_micron,
    temperature_K,
    luminosity_distance_m,
    redshift,
    vega_wavelength_micron,
    vega_flux_lambda,
):
    """
    WISE Vega-system catalog-equivalent flux density of ONE grain [Jy].

    This is an absolute physical single-grain calculation.  The target
    F_lambda is converted from SI W m^-3 to the CALSPEC convention
    erg s^-1 cm^-2 Angstrom^-1 before applying the same deterministic
    Vega-system band operator used elsewhere in this module.

    This function still does NOT assign the number of grains, dust mass,
    covering factor, or clump normalization.
    """
    lambda_obs_micron = np.asarray(
        bandpass.wavelength_micron,
        dtype=float,
    )

    F_lambda_si = grain_flux_lambda_observer_si(
        opacity=opacity,
        wavelength_obs_micron=lambda_obs_micron,
        grain_radius_micron=grain_radius_micron,
        temperature_K=temperature_K,
        luminosity_distance_m=luminosity_distance_m,
        redshift=redshift,
    )

    # W m^-3  ->  erg s^-1 cm^-2 Angstrom^-1
    #
    # 1 W = 1e7 erg/s
    # 1 m^-2 = 1e-4 cm^-2
    # per metre -> per Angstrom multiplies by 1e-10
    # Total factor = 1e7 * 1e-4 * 1e-10 = 1e-7.
    F_lambda_calspec = F_lambda_si * 1.0e-7

    numerator = np.trapezoid(
        F_lambda_calspec * np.asarray(bandpass.response, dtype=float),
        lambda_obs_micron,
    )

    scale_jy = wise_vega_scale_jy(
        bandpass,
        vega_wavelength_micron,
        vega_flux_lambda,
    )

    return float(numerator * scale_jy)
