"""
test_v4_wise_absolute_stage1.py

No UGC 11487 photometry.
No fitting.
No dust population normalization.

Tests only the absolute single-grain physics added to v4_wise_emission.py.
"""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v4_wise_emission import (
    grain_luminosity_lambda_si,
    grain_flux_lambda_observer_si,
)

SIGMA_SB = 5.670374419e-8
MICRON_M = 1e-6


class GreyOpacity:
    @staticmethod
    def q_abs(wavelength_micron, radius_micron):
        return np.ones_like(np.asarray(wavelength_micron, dtype=float))


def relerr(a, b):
    return abs(a-b)/abs(b)


print("="*88)
print("V4 ABSOLUTE SINGLE-GRAIN EMISSION TEST")
print("NO UGC 11487 DATA — NO FITTING — NO COVERING FACTOR")
print("="*88)

opacity = GreyOpacity()
a_um = 0.1
a_m = a_um * MICRON_M
T = 1000.0

# Wide enough wavelength range to recover the bolometric Planck integral.
lam_m = np.geomspace(1e-9, 1e-2, 50000)
lam_um = lam_m / MICRON_M

Llam = grain_luminosity_lambda_si(
    opacity,
    lam_um,
    a_um,
    T,
)

L_num = np.trapezoid(Llam, lam_m)
L_expected = 4.0*np.pi*a_m**2*SIGMA_SB*T**4
e = relerr(L_num, L_expected)

print(f"{'PASS' if e < 3e-4 else 'FAIL'} | grey grain bolometric luminosity")
print(f"       numerical = {L_num:.12e} W")
print(f"       expected  = {L_expected:.12e} W")
print(f"       rel.err   = {e:.3e}")

if e >= 3e-4:
    raise SystemExit(2)

# Inverse-square law at z=0.
lam_obs_um = np.geomspace(0.5, 30.0, 2000)
D1 = 1.0e20
D2 = 2.0e20

F1 = grain_flux_lambda_observer_si(
    opacity, lam_obs_um, a_um, T, D1, redshift=0.0
)
F2 = grain_flux_lambda_observer_si(
    opacity, lam_obs_um, a_um, T, D2, redshift=0.0
)

ratio = np.max(np.abs(F1/(F2*4.0) - 1.0))
print(f"{'PASS' if ratio < 1e-12 else 'FAIL'} | inverse-square distance scaling")
print(f"       max fractional mismatch = {ratio:.3e}")

if ratio >= 1e-12:
    raise SystemExit(2)

# Bolometric consistency at non-zero z:
# integrating F_lambda_obs over observed wavelength must give
# L_bol / (4*pi*D_L^2).
z = 0.2
D = 1.0e25
lam_rest_m = np.geomspace(1e-9, 1e-2, 50000)
lam_obs_m = lam_rest_m*(1.0+z)
lam_obs_um = lam_obs_m/MICRON_M

Fobs = grain_flux_lambda_observer_si(
    opacity, lam_obs_um, a_um, T, D, redshift=z
)

Fbol_num = np.trapezoid(Fobs, lam_obs_m)
Fbol_expected = L_expected/(4.0*np.pi*D**2)
e2 = relerr(Fbol_num, Fbol_expected)

print(f"{'PASS' if e2 < 3e-4 else 'FAIL'} | redshift + luminosity-distance bolometric closure")
print(f"       numerical = {Fbol_num:.12e} W m^-2")
print(f"       expected  = {Fbol_expected:.12e} W m^-2")
print(f"       rel.err   = {e2:.3e}")

if e2 >= 3e-4:
    raise SystemExit(2)

print("="*88)
print("ALL ABSOLUTE SINGLE-GRAIN TESTS PASSED.")
print("This validates units and cosmological spectral mapping only.")
print("Grain number / dust mass / covering factor remain intentionally undefined.")
print("="*88)
