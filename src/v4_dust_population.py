"""
v4_dust_population.py
=====================

Physical normalization of a grain population for the UGC 11487 V4 model.

NO WISE DATA.
NO FITTING.
NO EMPIRICAL MIR AMPLITUDE.

This module defines an effective initial absorbed fraction

    f_abs,0 = sum_i C_abs,src,i / (4*pi*r_i^2)

for a population of independent, non-overlapping absorbers illuminated by
an isotropic central source.

For a spherical grain

    C_abs,src(a) = pi*a^2*<Qabs>_src(a),

where <Qabs>_src is taken directly from the same DustThermalEquilibrium
object used by the radiative-equilibrium temperature solver.

IMPORTANT
---------
f_abs,0 is an effective absorption fraction in this single-interception,
optically-thin/non-overlap normalization.  It must NOT automatically be
identified with the geometric covering factor of an optically thick,
self-obscuring clumpy torus.

The module only determines the total number of grains corresponding to a
chosen f_abs,0.  It does not choose or fit f_abs,0.
"""

from dataclasses import dataclass
import numpy as np

from src.v4_wise_emission import grain_size_number_weights

MICRON_M = 1.0e-6
PC_M = 3.085677581491367e16


@dataclass
class DustPopulationNormalization:
    f_abs_initial: float
    total_grain_number: float
    mean_source_absorption_cross_section_m2: float
    mean_inverse_4pi_r2_m2inv: float
    grain_radius_micron: np.ndarray
    grain_number_weights: np.ndarray
    source_mean_qabs: np.ndarray


def source_absorption_cross_section_m2(
    thermal_solver,
    grain_radius_micron,
):
    """
    Source-SED-weighted absorption cross section of each grain size.

        C_abs,src = pi a^2 <Qabs>_src

    Returns
    -------
    ndarray, shape (Na,), in m^2.
    """
    a_um = np.asarray(grain_radius_micron, dtype=float)

    if a_um.ndim != 1 or a_um.size < 2:
        raise ValueError("grain_radius_micron must be a 1D grid with >=2 sizes.")
    if np.any(a_um <= 0.0) or np.any(np.diff(a_um) <= 0.0):
        raise ValueError("grain sizes must be positive and strictly increasing.")

    qsrc = np.asarray(
        thermal_solver.source_mean_qabs(a_um),
        dtype=float,
    )

    if qsrc.shape != a_um.shape:
        raise RuntimeError("source_mean_qabs returned an unexpected shape.")
    if np.any(~np.isfinite(qsrc)) or np.any(qsrc < 0.0):
        raise RuntimeError("Invalid source-mean Qabs.")

    a_m = a_um * MICRON_M
    return np.pi * a_m**2 * qsrc


def build_population_normalization(
    thermal_solver,
    grain_radius_micron,
    clump_radius_pc,
    f_abs_initial,
    q=3.5,
    clump_weight=None,
):
    """
    Determine total grain number from a specified initial absorbed fraction.

    The joint distribution is assumed separable:
      - grain-size number PDF: dn/da proportional to a^(-q);
      - spatial number PDF: represented by clump radii and clump weights.

    With normalized size weights w_a and spatial weights w_r,

      f_abs,0 =
          N_total *
          [sum_a w_a C_abs,src(a)] *
          [sum_r w_r / (4*pi*r^2)].

    Therefore N_total is determined, not fitted internally.
    """
    f_abs = float(f_abs_initial)

    if not np.isfinite(f_abs) or f_abs <= 0.0 or f_abs > 1.0:
        raise ValueError("f_abs_initial must satisfy 0 < f_abs_initial <= 1.")

    grains = np.asarray(grain_radius_micron, dtype=float)
    radii_pc = np.asarray(clump_radius_pc, dtype=float)

    if radii_pc.ndim != 1 or radii_pc.size == 0:
        raise ValueError("clump_radius_pc must be a non-empty 1D array.")
    if np.any(~np.isfinite(radii_pc)) or np.any(radii_pc <= 0.0):
        raise ValueError("clump radii must be finite and positive.")

    if clump_weight is None:
        wr = np.full(radii_pc.size, 1.0 / radii_pc.size, dtype=float)
    else:
        wr = np.asarray(clump_weight, dtype=float)
        if wr.shape != radii_pc.shape:
            raise ValueError("clump_weight shape mismatch.")
        if np.any(~np.isfinite(wr)) or np.any(wr < 0.0):
            raise ValueError("Invalid clump weights.")
        sw = float(np.sum(wr))
        if sw <= 0.0:
            raise ValueError("clump weights must have positive sum.")
        wr = wr / sw

    wa = grain_size_number_weights(grains, q=q)

    cabs = source_absorption_cross_section_m2(
        thermal_solver=thermal_solver,
        grain_radius_micron=grains,
    )

    qsrc = np.asarray(
        thermal_solver.source_mean_qabs(grains),
        dtype=float,
    )

    mean_cabs = float(np.sum(wa * cabs))

    r_m = radii_pc * PC_M
    mean_inv = float(
        np.sum(wr / (4.0 * np.pi * r_m**2))
    )

    denominator = mean_cabs * mean_inv

    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("Population absorption normalization is non-positive.")

    n_total = f_abs / denominator

    return DustPopulationNormalization(
        f_abs_initial=f_abs,
        total_grain_number=float(n_total),
        mean_source_absorption_cross_section_m2=mean_cabs,
        mean_inverse_4pi_r2_m2inv=mean_inv,
        grain_radius_micron=grains.copy(),
        grain_number_weights=wa.copy(),
        source_mean_qabs=qsrc.copy(),
    )


def recovered_initial_absorbed_fraction(
    normalization,
):
    """
    Reconstruct f_abs,0 from the stored physical factors.
    """
    return float(
        normalization.total_grain_number
        * normalization.mean_source_absorption_cross_section_m2
        * normalization.mean_inverse_4pi_r2_m2inv
    )


def surviving_absorbed_fraction(
    normalization,
    clump_radius_pc,
    alive,
    clump_weight=None,
):
    """
    Effective absorbed fraction after sublimation.

    Parameters
    ----------
    alive : ndarray
        Shape (..., Nclump, Na).  False grains are removed permanently.

    Notes
    -----
    The original total grain number is retained.  Surviving grains are NOT
    renormalized after sublimation.
    """
    radii_pc = np.asarray(clump_radius_pc, dtype=float)
    alive = np.asarray(alive, dtype=bool)

    if alive.shape[-2:] != (
        radii_pc.size,
        normalization.grain_radius_micron.size,
    ):
        raise ValueError(
            "alive must end with dimensions (Nclump, Ngrain_size)."
        )

    if clump_weight is None:
        wr = np.full(radii_pc.size, 1.0 / radii_pc.size, dtype=float)
    else:
        wr = np.asarray(clump_weight, dtype=float)
        if wr.shape != radii_pc.shape:
            raise ValueError("clump_weight shape mismatch.")
        if np.any(~np.isfinite(wr)) or np.any(wr < 0.0):
            raise ValueError("Invalid clump weights.")
        sw = float(np.sum(wr))
        if sw <= 0.0:
            raise ValueError("clump weights must have positive sum.")
        wr = wr / sw

    wa = normalization.grain_number_weights

    a_m = normalization.grain_radius_micron * MICRON_M
    cabs = (
        np.pi
        * a_m**2
        * normalization.source_mean_qabs
    )

    r_m = radii_pc * PC_M
    geometric = wr / (4.0 * np.pi * r_m**2)

    per_cell = (
        geometric[:, None]
        * wa[None, :]
        * cabs[None, :]
    )

    return (
        normalization.total_grain_number
        * np.sum(
            alive * per_cell,
            axis=(-2, -1),
        )
    )
