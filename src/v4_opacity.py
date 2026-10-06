"""
v4_opacity.py
=============

Dust optical-property layer for the physical UGC 11487 V4 model.

Reads the B.T. Draine / Laor & Draine optical-property tables:

    Sil_81.gz : astronomical silicate
    Gra_81.gz : graphite, 1/3-2/3 approximation

Native tables:
    81 grain radii
    241 wavelengths per radius
    wavelength [micron]
    Q_abs
    Q_sca
    g=<cos>

The primary quantity used by the thermal solver is

    Q_abs(lambda, a, composition).

Interpolation is performed in log(lambda), log(a), and log(Q_abs)
where Q_abs is positive.

No WISE data, dust temperatures, or fitted quantities occur here.
"""

from pathlib import Path
import gzip

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ============================================================
# Supported compositions
# ============================================================

SUPPORTED_COMPOSITIONS = {
    "silicate": "Sil_81.gz",
    "graphite": "Gra_81.gz",
}


# ============================================================
# Raw Draine parser
# ============================================================

def read_draine_table(path):
    """
    Read a Draine 81-radius optical-property table.

    Returns
    -------
    dict with arrays:

        radius_micron : (NRAD,)
        wavelength_micron : (NWAV,)
        q_abs : (NRAD, NWAV)
        q_sca : (NRAD, NWAV)
        g     : (NRAD, NWAV)

    Wavelength is returned in ascending order.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as f:
        lines = f.readlines()

    if len(lines) < 10:
        raise RuntimeError(
            f"Draine table appears incomplete: {path}"
        )

    # Header explicitly gives NRAD and NWAV.
    try:
        nrad = int(lines[3].split()[0])
        nwav = int(lines[4].split()[0])
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse Draine header: {path}"
        ) from exc

    radii = []
    all_wave = []
    all_qabs = []
    all_qsca = []
    all_g = []

    i = 6

    while i < len(lines):

        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if "radius(micron)" not in line.lower():
            i += 1
            continue

        # Example:
        # 1.000E-03 = radius(micron) Astronomical silicate
        try:
            radius = float(
                line.split("=")[0].strip()
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Could not parse grain radius at line {i+1}"
            ) from exc

        # Next line is the column header.
        i += 2

        rows = []

        for _ in range(nwav):

            if i >= len(lines):
                raise RuntimeError(
                    "Unexpected end of Draine file."
                )

            parts = lines[i].split()

            if len(parts) < 4:
                raise RuntimeError(
                    f"Invalid optical-property row at line {i+1}: "
                    f"{lines[i]!r}"
                )

            try:
                values = [
                    float(parts[j].replace("D", "E"))
                    for j in range(4)
                ]
            except ValueError as exc:
                raise RuntimeError(
                    f"Non-numeric optical-property row "
                    f"at line {i+1}"
                ) from exc

            rows.append(values)
            i += 1

        arr = np.asarray(
            rows,
            dtype=float,
        )

        wave = arr[:, 0]
        qabs = arr[:, 1]
        qsca = arr[:, 2]
        gg = arr[:, 3]

        # Draine wavelength is descending.
        order = np.argsort(wave)

        wave = wave[order]
        qabs = qabs[order]
        qsca = qsca[order]
        gg = gg[order]

        radii.append(radius)
        all_wave.append(wave)
        all_qabs.append(qabs)
        all_qsca.append(qsca)
        all_g.append(gg)

    radii = np.asarray(
        radii,
        dtype=float,
    )

    if len(radii) != nrad:
        raise RuntimeError(
            f"Expected {nrad} radius blocks but found "
            f"{len(radii)} in {path}"
        )

    # Verify identical wavelength grids.
    reference_wave = all_wave[0]

    for j, wave in enumerate(all_wave[1:], start=1):

        if not np.allclose(
            wave,
            reference_wave,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"Wavelength grid differs for radius block {j}."
            )

    q_abs = np.asarray(
        all_qabs,
        dtype=float,
    )

    q_sca = np.asarray(
        all_qsca,
        dtype=float,
    )

    g = np.asarray(
        all_g,
        dtype=float,
    )

    # Radius order should already be ascending,
    # but enforce it explicitly.
    rorder = np.argsort(radii)

    radii = radii[rorder]
    q_abs = q_abs[rorder]
    q_sca = q_sca[rorder]
    g = g[rorder]

    return {
        "path": str(path),
        "n_radius": int(nrad),
        "n_wavelength": int(nwav),
        "radius_micron": radii,
        "wavelength_micron": reference_wave,
        "q_abs": q_abs,
        "q_sca": q_sca,
        "g": g,
    }


# ============================================================
# Qabs interpolator
# ============================================================

class DustOpacity:
    """
    Interpolator for Draine Q_abs(lambda, a).

    Interpolation coordinates:

        log10(a)
        log10(lambda)

    and the interpolated value is

        log10(Q_abs).

    This is preferable to linear interpolation because the tables
    span many orders of magnitude.
    """

    def __init__(
        self,
        table,
        composition,
    ):

        self.table = table
        self.composition = str(composition)

        self.radius_micron = np.asarray(
            table["radius_micron"],
            dtype=float,
        )

        self.wavelength_micron = np.asarray(
            table["wavelength_micron"],
            dtype=float,
        )

        self.q_abs_grid = np.asarray(
            table["q_abs"],
            dtype=float,
        )

        if np.any(self.radius_micron <= 0):
            raise ValueError(
                "Grain radii must be positive."
            )

        if np.any(self.wavelength_micron <= 0):
            raise ValueError(
                "Wavelengths must be positive."
            )

        if np.any(self.q_abs_grid <= 0):
            raise ValueError(
                "Q_abs must be positive for logarithmic interpolation."
            )

        self.log_a = np.log10(
            self.radius_micron
        )

        self.log_lambda = np.log10(
            self.wavelength_micron
        )

        self.log_qabs = np.log10(
            self.q_abs_grid
        )

        self._interpolator = RegularGridInterpolator(
            (
                self.log_a,
                self.log_lambda,
            ),
            self.log_qabs,
            method="linear",
            bounds_error=True,
        )

    # --------------------------------------------------------
    # Limits
    # --------------------------------------------------------

    @property
    def a_min_micron(self):
        return float(
            self.radius_micron[0]
        )

    @property
    def a_max_micron(self):
        return float(
            self.radius_micron[-1]
        )

    @property
    def lambda_min_micron(self):
        return float(
            self.wavelength_micron[0]
        )

    @property
    def lambda_max_micron(self):
        return float(
            self.wavelength_micron[-1]
        )

    # --------------------------------------------------------
    # Interpolation
    # --------------------------------------------------------

    def q_abs(
        self,
        wavelength_micron,
        radius_micron,
    ):
        """
        Evaluate Q_abs(lambda, a).

        Broadcasting is supported.

        Examples
        --------
        q_abs(4.6, 0.1)

        q_abs(
            wavelength_array[:, None],
            radius_array[None, :]
        )
        """

        lam = np.asarray(
            wavelength_micron,
            dtype=float,
        )

        a = np.asarray(
            radius_micron,
            dtype=float,
        )

        if np.any(lam <= 0):
            raise ValueError(
                "Wavelength must be positive."
            )

        if np.any(a <= 0):
            raise ValueError(
                "Grain radius must be positive."
            )

        lam_b, a_b = np.broadcast_arrays(
            lam,
            a,
        )

        if (
            np.any(a_b < self.a_min_micron)
            or np.any(a_b > self.a_max_micron)
        ):
            raise ValueError(
                f"Requested grain radius outside "
                f"Draine range "
                f"[{self.a_min_micron}, "
                f"{self.a_max_micron}] micron."
            )

        if (
            np.any(lam_b < self.lambda_min_micron)
            or np.any(lam_b > self.lambda_max_micron)
        ):
            raise ValueError(
                f"Requested wavelength outside "
                f"Draine range "
                f"[{self.lambda_min_micron}, "
                f"{self.lambda_max_micron}] micron."
            )

        points = np.column_stack(
            (
                np.log10(a_b.ravel()),
                np.log10(lam_b.ravel()),
            )
        )

        log_q = self._interpolator(
            points
        )

        q = 10.0**log_q

        q = q.reshape(
            lam_b.shape
        )

        if q.ndim == 0:
            return float(q)

        return q


# ============================================================
# Library loader
# ============================================================

def load_dust_opacity(
    reference_dir,
    composition,
):
    """
    Load one supported Draine dust composition.
    """

    reference_dir = Path(
        reference_dir
    )

    composition = str(
        composition
    ).lower()

    if composition not in SUPPORTED_COMPOSITIONS:
        raise ValueError(
            f"Unsupported composition: {composition}. "
            f"Available: "
            f"{list(SUPPORTED_COMPOSITIONS)}"
        )

    path = (
        reference_dir
        / SUPPORTED_COMPOSITIONS[composition]
    )

    table = read_draine_table(
        path
    )

    return DustOpacity(
        table=table,
        composition=composition,
    )


# ============================================================
# Grain-size distributions
# ============================================================

def make_grain_size_grid(
    a_min_micron,
    a_max_micron,
    n_size=41,
):
    """
    Logarithmically spaced grain-size grid.
    """

    a_min = float(a_min_micron)
    a_max = float(a_max_micron)
    n_size = int(n_size)

    if a_min <= 0:
        raise ValueError(
            "a_min_micron must be positive."
        )

    if a_max <= a_min:
        raise ValueError(
            "a_max_micron must exceed a_min_micron."
        )

    if n_size < 2:
        raise ValueError(
            "n_size must be >= 2."
        )

    return np.geomspace(
        a_min,
        a_max,
        n_size,
    )


def grain_number_weights(
    radius_micron,
    q=3.5,
):
    """
    Numerical integration weights for

        dn/da proportional to a^(-q).

    These are NUMBER weights.

    They are normalized so that

        sum(weights) = 1.

    The thermal/emission layer must still include the grain
    cross-sectional area proportional to a^2.
    """

    a = np.asarray(
        radius_micron,
        dtype=float,
    )

    if a.ndim != 1:
        raise ValueError(
            "radius_micron must be one-dimensional."
        )

    if len(a) < 2:
        raise ValueError(
            "At least two grain sizes are required."
        )

    if np.any(a <= 0):
        raise ValueError(
            "Grain radii must be positive."
        )

    if np.any(np.diff(a) <= 0):
        raise ValueError(
            "Grain-size grid must be increasing."
        )

    # Cell boundaries in logarithmic space.
    edges = np.empty(
        len(a) + 1,
        dtype=float,
    )

    edges[1:-1] = np.sqrt(
        a[:-1] * a[1:]
    )

    edges[0] = (
        a[0]**2 / edges[1]
    )

    edges[-1] = (
        a[-1]**2 / edges[-2]
    )

    q = float(q)

    if abs(q - 1.0) < 1.0e-12:

        w = np.log(
            edges[1:] / edges[:-1]
        )

    else:

        power = 1.0 - q

        w = (
            edges[1:]**power
            - edges[:-1]**power
        ) / power

    if np.any(w <= 0):
        raise RuntimeError(
            "Invalid grain-size integration weights."
        )

    w /= np.sum(w)

    return w