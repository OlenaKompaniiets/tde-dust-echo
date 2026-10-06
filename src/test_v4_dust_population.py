"""
test_v4_dust_population.py
==========================

Independent physical-normalization tests.

NO UGC 11487 DATA.
NO FITTING.
NO WISE LIGHT CURVE.
"""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v4_dust_population import (
    build_population_normalization,
    recovered_initial_absorbed_fraction,
    surviving_absorbed_fraction,
)


class ControlledThermalSolver:
    """
    Test double with known <Qabs>_src.
    This isolates the population-normalization algebra.
    """
    @staticmethod
    def source_mean_qabs(radius_micron):
        a = np.asarray(radius_micron, dtype=float)
        return 0.35 + 0.20 * np.log10(a / np.min(a) + 1.0)


print("=" * 88)
print("V4 DUST POPULATION NORMALIZATION TEST")
print("NO UGC 11487 DATA — NO FITTING")
print("=" * 88)

solver = ControlledThermalSolver()

grains = np.geomspace(0.01, 1.0, 41)
radii = np.array([0.2, 0.4, 0.8, 1.6, 3.2])
f_target = 0.173

norm = build_population_normalization(
    thermal_solver=solver,
    grain_radius_micron=grains,
    clump_radius_pc=radii,
    f_abs_initial=f_target,
    q=3.5,
)

f_recovered = recovered_initial_absorbed_fraction(norm)
err = abs(f_recovered - f_target)

print(f"{'PASS' if err < 1e-14 else 'FAIL'} | exact initial absorption closure")
print(f"       target    = {f_target:.15f}")
print(f"       recovered = {f_recovered:.15f}")
print(f"       abs.err   = {err:.3e}")

if err >= 1e-14:
    raise SystemExit(2)

# All grains alive must recover the same initial fraction.
alive_all = np.ones((radii.size, grains.size), dtype=bool)
f_all = float(surviving_absorbed_fraction(norm, radii, alive_all))
err_all = abs(f_all - f_target)

print(f"{'PASS' if err_all < 1e-14 else 'FAIL'} | all-alive survival closure")
print(f"       recovered = {f_all:.15f}")

if err_all >= 1e-14:
    raise SystemExit(2)

# No grains alive -> zero absorption.
alive_none = np.zeros_like(alive_all)
f_none = float(surviving_absorbed_fraction(norm, radii, alive_none))

print(f"{'PASS' if f_none == 0.0 else 'FAIL'} | complete destruction -> zero absorption")
print(f"       recovered = {f_none:.15f}")

if f_none != 0.0:
    raise SystemExit(2)

# Destroy an increasing fraction of cells. Effective absorption must never rise.
series = []
for k in range(radii.size + 1):
    alive = np.ones_like(alive_all)
    if k > 0:
        alive[:k, :] = False
    series.append(float(surviving_absorbed_fraction(norm, radii, alive)))

series = np.asarray(series)
monotonic = np.all(np.diff(series) <= 1e-15)

print(f"{'PASS' if monotonic else 'FAIL'} | sublimation cannot increase absorbed fraction")
print("       sequence =", np.array2string(series, precision=8))

if not monotonic:
    raise SystemExit(2)

# Scaling f_abs by a factor must scale N_total by the same factor.
norm2 = build_population_normalization(
    thermal_solver=solver,
    grain_radius_micron=grains,
    clump_radius_pc=radii,
    f_abs_initial=2.0 * f_target,
    q=3.5,
)

ratio = norm2.total_grain_number / norm.total_grain_number
err_ratio = abs(ratio - 2.0)

print(f"{'PASS' if err_ratio < 1e-14 else 'FAIL'} | linear grain-number scaling")
print(f"       N2/N1 = {ratio:.15f}")

if err_ratio >= 1e-14:
    raise SystemExit(2)

print("=" * 88)
print("ALL POPULATION-NORMALIZATION TESTS PASSED.")
print("This validates the independent/non-overlapping absorption normalization only.")
print("It does NOT establish a geometric covering factor for an optically thick torus.")
print("=" * 88)
