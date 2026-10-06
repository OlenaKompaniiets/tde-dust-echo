"""Audit V4.6 energy bookkeeping without modifying validated V4.6 code.

Run from project root:
    python scripts\audit_v46_energy.py

This script calls the existing scripts.run_v46_ir_reprocessing.simulate()
for the fiducial ir_fraction=0.5 and independently integrates the energy
quantities returned by the validated V4.6 implementation.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_v46_ir_reprocessing as v46


def integ_1d(power, t_days):
    return float(np.trapezoid(np.asarray(power, float), np.asarray(t_days, float)) * 86400.0)


def integ_cells(power, t_days):
    arr = np.asarray(power, float)
    return np.trapezoid(arr, np.asarray(t_days, float), axis=0) * 86400.0


def main():
    c = dict(v46.BASE, ir_fraction=0.5)

    # Reproduce the exact driver used by the V4.6 runner.
    t, L, fullE = v46.v45.driver(
        c["stars"], c["tau_visc"], c["eta"], c["dt"], c["source_end_days"]
    )

    # Reconstruct the exact V4.6 transfer call, retaining its raw bookkeeping arrays.
    op, th, a, lookup = v46.g._cached_dust_static(
        c["composition"], c["amin"], c["amax"]
    )
    wa = v46.grain_size_number_weights(a, q=c["q"])
    qs = th.source_mean_qabs(a)
    lam = th.wavelength_m
    C = (
        np.pi * (a * 1e-6) ** 2
    )[:, None] * op.q_abs(th.wavelength_micron[None, :], a[:, None])

    f = float(np.cos(np.deg2rad(c["theta"])))

    radii = []
    nums = []
    for key, tau in (("compact", c["tau_C"]), ("extended", c["tau_E"])):
        r, rw = v46.g.radial_grid_and_weights(*c[key], c["n_radii"])
        pop = v46.g.build_population_normalization(
            th, a, r, 1.0, q=c["q"], clump_weight=rw
        )
        N = pop.total_grain_number * f * tau
        radii.append(r)
        nums.append(N * rw[:, None] * wa[None, :])

    radii_all = np.concatenate(radii)
    numbers = np.concatenate(nums)
    sub = float(v46.g.sublimation_temperature_K(c["density"], c["composition"]))

    z = v46.solve_direct_plus_ir(
        t, L, lam, th.source_sed, C, qs, lookup,
        radii_all, numbers, f, sub,
        attenuation=True,
        ir_reprocess_fraction=0.5,
    )

    nr_c = len(radii[0])
    nr_e = len(radii[1])

    Edir_cells = integ_cells(z["absorbed_direct_erg_s"], t)
    Eir_cells = integ_cells(z["absorbed_ir_erg_s"], t)
    Eemit_cells = integ_cells(z["emitted_dust_erg_s"], t)

    sl_c = slice(0, nr_c)
    sl_e = slice(nr_c, nr_c + nr_e)

    Esource = integ_1d(L, t)
    E_direct_C = float(Edir_cells[sl_c].sum())
    E_direct_E = float(Edir_cells[sl_e].sum())
    E_secondary_C = float(Eir_cells[sl_c].sum())
    E_secondary_E = float(Eir_cells[sl_e].sum())
    E_emit_C = float(Eemit_cells[sl_c].sum())
    E_emit_E = float(Eemit_cells[sl_e].sum())

    E_inner_escape = integ_1d(z["escaped_ir_inner_erg_s"], t)
    E_outer_escape = integ_1d(z["escaped_ir_outer_erg_s"], t)
    E_direct_escape = integ_1d(z["escaped_direct_erg_s"], t)
    E_total_escape = integ_1d(z["escaped_total_erg_s"], t)

    # In the present 1-D outward-stream approximation, compact emitted energy
    # launched outward is eps * E_emit_C. This is the energy available to illuminate
    # later cells before attenuation there.
    eps = float(z["ir_reprocess_fraction"])
    E_IR_C_outward_launched = eps * E_emit_C

    print("=" * 78)
    print("V4.6 ENERGY AUDIT -- ir_fraction = 0.5")
    print("=" * 78)
    print(f"E_source(window)             = {Esource:.12e} erg")
    print(f"E_source(full driver)        = {float(fullE):.12e} erg")
    print()
    print(f"E_direct_absorbed_compact    = {E_direct_C:.12e} erg")
    print(f"E_direct_absorbed_extended   = {E_direct_E:.12e} erg")
    print(f"E_secondary_abs_compact      = {E_secondary_C:.12e} erg")
    print(f"E_secondary_abs_extended     = {E_secondary_E:.12e} erg")
    print()
    print(f"E_dust_emitted_compact       = {E_emit_C:.12e} erg")
    print(f"E_dust_emitted_extended      = {E_emit_E:.12e} erg")
    print(f"E_IR_C_outward_launched      = {E_IR_C_outward_launched:.12e} erg")
    print()
    print(f"E_direct_escape              = {E_direct_escape:.12e} erg")
    print(f"E_IR_inner_escape            = {E_inner_escape:.12e} erg")
    print(f"E_IR_outer_escape            = {E_outer_escape:.12e} erg")
    print(f"E_total_escape               = {E_total_escape:.12e} erg")
    print()
    Edir = E_direct_C + E_direct_E
    Esec = E_secondary_C + E_secondary_E
    print("--- RATIOS ---")
    print(f"secondary/direct absorbed    = {Esec / max(Edir, 1e-300):.12e}")
    print(f"secondary_E/direct_E         = {E_secondary_E / max(E_direct_E, 1e-300):.12e}")
    print(f"secondary_E/C-outward-launch = {E_secondary_E / max(E_IR_C_outward_launched, 1e-300):.12e}")
    print(f"C-outward/source             = {E_IR_C_outward_launched / max(Esource, 1e-300):.12e}")
    print()
    print("--- CLOSURE ---")
    print(f"(E_total_escape-E_source)/E_source = {(E_total_escape-Esource)/max(Esource,1e-300):.12e}")
    print(f"reported instantaneous max error    = {z['energy_budget_relative_max']:.12e}")
    print()
    print("--- INITIAL RADIAL TAU ---")
    for wave in (3.4, 4.6, 10.0):
        tau = float(np.interp(wave * 1e-6, lam, z["initial_tau_lambda"]))
        print(f"tau({wave:4.1f} um) = {tau:.8f}")

    # Critical consistency diagnostics.
    print()
    print("--- CONSISTENCY CHECKS ---")
    print(f"sum emitted / sum absorbed = {(E_emit_C+E_emit_E)/max(Edir+Esec,1e-300):.12e}")
    if Esec / max(Edir, 1e-300) < 1e-8:
        print("WARNING: integrated secondary absorption is negligible (<1e-8 of direct).")
        print("If W1/W2 changes strongly relative to ir_fraction=0, the photometric")
        print("response is not being normalized consistently with the bolometric transfer.")
    else:
        print("Secondary absorption is energetically non-negligible.")

    # Also reproduce the actual fit number using the unchanged runner.
    fit = v46.evaluate(c)
    print()
    print("--- FIT REPRODUCTION ---")
    print(f"chi2_total = {fit['chi2_total']:.12f}")
    print(f"chi2_W1    = {fit['chi2_w1']:.12f}")
    print(f"chi2_W2    = {fit['chi2_w2']:.12f}")
    print(f"t0_MJD     = {fit['t0_mjd']:.8f}")


if __name__ == "__main__":
    main()
