"""
UGC 11487 V4.1 — minimal structured two-population dust experiment.

Scientific question
-------------------
Can two *physical* radial dust populations, illuminated by the same STARS/TDE
driver and passed through the already validated V4 thermal/sublimation/WISE/3-D
echo machinery, reproduce the 22-epoch W1/W2 data substantially better than
the single-population V4 family?

This is deliberately a minimal V4.1 experiment:
  * same driver for both populations;
  * graphite only in this first test;
  * same grain-size law in both populations;
  * no fitted dust temperatures or lags;
  * no W1/W2-specific amplitudes;
  * each population has one common physical f_abs normalization affecting both
    W1 and W2;
  * compact and extended radial populations are non-overlapping by construction;
  * observational likelihood is still the direct 44-point W1/W2 chi-square.

The two absorbed-fraction coefficients are solved as non-negative linear
coefficients at each t0.  An optional physical cap f_comp + f_ext <= 0.8 is
enforced.  This cap is an experiment boundary, not an observational penalty.

Run from project root:
    python scripts\\run_v41_structured_test.py

Output:
    results\\v41_structured_test\\v41_structured_results.csv
    results\\v41_structured_test\\v41_best_epoch_table.csv
    results\\v41_structured_test\\v41_best_residuals.png
    results\\v41_structured_test\\v41_manifest.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the validated fixed-V4 implementation directly.
import scripts.run_v4_model_grid as g


OUTDIR = PROJECT_ROOT / "results" / "v41_structured_test"
OUTCSV = OUTDIR / "v41_structured_results.csv"

# ---------------------------------------------------------------------------
# Minimal V4.1 scientific experiment
# ---------------------------------------------------------------------------
# Start with the driver/grain physics of the current fixed-V4 global control
# minimum.  The purpose here is NOT global inference: it is a controlled test
# of whether radial structure alone removes the dominant residual topology.
STARS_REL = "input/m0.3_t0.0/0.900.dat"
ETA_RAD = 0.10
TAU_VISC_DAYS = 0.0
COMPOSITION = "graphite"
DENSITY_CM3 = 1.0e9
GRAIN_CONFIG = (0.010, 1.00, 3.5)
THETA_OPEN_DEG = 60.0
INCLINATION_DEG = 55.0

# Compact and extended populations are non-overlapping.
# These boundaries are intentionally coarse for the first falsification test.
COMPACT_ZONES = (
    (0.05, 0.15),
    (0.05, 0.30),
    (0.10, 0.30),
    (0.15, 0.40),
)
EXTENDED_ZONES = (
    (0.30, 1.00),
    (0.30, 2.00),
    (0.40, 1.50),
    (0.50, 3.00),
    (0.75, 3.00),
)
P_VALUES = (0.0, 1.0, 2.0)

# t0 remains a common source-time translation for both populations.
T0_GRID = np.arange(56000.0, 58000.1, 20.0)

# Effective absorbed fractions of the two non-overlapping populations.
# They are solved continuously (not gridded) because the model is exactly
# linear in grain number after the thermal calculation.
FABS_COMPONENT_MAX = 0.80
FABS_TOTAL_MAX = 0.80


def weighted_nonnegative_two_component_fit(y1, y2, s1, s2,
                                           base1, base2,
                                           e1c, e2c, e1e, e2e):
    """
    Solve the two physical population normalizations for one common t0.

    Design matrix columns are compact and extended dust populations.
    Each coefficient scales W1 and W2 together, so there is no independent
    band amplitude.

    First solve bounded NNLS. If the total absorbed fraction exceeds the
    experiment cap, solve exactly on f_comp + f_ext = FABS_TOTAL_MAX by a
    one-dimensional weighted least-squares projection.
    """
    y = np.concatenate(((y1 - base1) / s1, (y2 - base2) / s2))
    A = np.column_stack((
        np.concatenate((e1c / s1, e2c / s2)),
        np.concatenate((e1e / s1, e2e / s2)),
    ))

    sol = lsq_linear(
        A, y,
        bounds=(np.zeros(2), np.full(2, FABS_COMPONENT_MAX)),
        method="trf", lsmr_tol="auto"
    )
    f = np.asarray(sol.x, float)

    if f.sum() > FABS_TOTAL_MAX:
        # Constrained boundary: f_e = Ftot - f_c.
        # Minimize || y - A_e*Ftot - (A_c-A_e)*f_c ||^2.
        d = A[:, 0] - A[:, 1]
        y0 = y - A[:, 1] * FABS_TOTAL_MAX
        den = float(np.dot(d, d))
        fc = 0.5 * FABS_TOTAL_MAX if den <= 0 else float(np.dot(d, y0) / den)
        fc = float(np.clip(fc, 0.0, FABS_TOTAL_MAX))
        f = np.array([fc, FABS_TOTAL_MAX - fc], float)

    model1 = base1 + f[0] * e1c + f[1] * e1e
    model2 = base2 + f[0] * e2c + f[1] * e2e
    r1 = (y1 - model1) / s1
    r2 = (y2 - model2) / s2
    return f, model1, model2, float(np.sum(r1*r1)), float(np.sum(r2*r2))


@lru_cache(maxsize=None)
def build_population_echo(rin, rout, p):
    """
    Build f_abs=1 observer-frame W1/W2 echoes for one radial population using
    exactly the validated fixed-V4 physical machinery.
    """
    stars_file = g.STARS_ROOT / STARS_REL
    if not stars_file.exists():
        raise FileNotFoundError(f"STARS curve not found: {stars_file}")

    driver, diag = g.build_physical_driver(
        stars_file=stars_file,
        M_BH_Msun=g.M_BH_MSUN,
        eta_rad=ETA_RAD,
        tau_visc_days=TAU_VISC_DAYS,
        t_circ_days=g.T_CIRC_DAYS,
        dt_days=g.DRIVER_DT_DAYS,
        pre_peak_days=g.PRE_PEAK_DAYS,
        post_fallback_days=g.POST_FALLBACK_DAYS,
    )
    td = driver["t_days"].to_numpy(float)
    L = driver["L_bol_erg_s"].to_numpy(float)

    amin, amax, q = GRAIN_CONFIG
    opacity, thermal, grains, lookup = g._cached_dust_static(
        COMPOSITION, amin, amax
    )
    radii, rw = g.radial_grid_and_weights(float(rin), float(rout), float(p), g.N_RADII)

    dust = g._build_dust_response_cached(
        time_days=td,
        luminosity_erg_s=L,
        radius_pc=radii,
        grains=grains,
        lookup=lookup,
        composition=COMPOSITION,
        density_cm3=DENSITY_CM3,
    )

    pop = g.build_population_normalization(
        thermal_solver=thermal,
        grain_radius_micron=grains,
        clump_radius_pc=radii,
        f_abs_initial=1.0,
        q=q,
        clump_weight=rw,
    )

    w1, w2, vw, vf = g._cached_wise_static()
    dl_m = g.D_L_MPC * 1.0e6 * g.PC_M

    pergrain_w1 = g.absolute_population_band_cube(
        opacity, w1, grains, dust.temperature_K, dust.alive, q, g.Z,
        dl_m, vw, vf, g.WISE_TGRID,
    )
    pergrain_w2 = g.absolute_population_band_cube(
        opacity, w2, grains, dust.temperature_K, dust.alive, q, g.Z,
        dl_m, vw, vf, g.WISE_TGRID,
    )
    radial_w1 = pop.total_grain_number * pergrain_w1
    radial_w2 = pop.total_grain_number * pergrain_w2

    rc, zc = g._cached_geometry(
        float(rin), float(rout), float(p), THETA_OPEN_DEG, INCLINATION_DEG
    )
    e1, e2 = g.integrate_radial_responses_over_clumps(
        dust_time_rest_days=td,
        radius_grid_pc=radii,
        radial_responses=np.stack((radial_w1, radial_w2), axis=0),
        clump_radius_pc=rc,
        clump_z_obs_pc=zc,
        redshift=g.Z,
        chunk_size=256,
        source_time_input=True,
    )

    return {
        "time": np.asarray(e1.time_obs_days, float),
        "w1": np.asarray(e1.response, float),
        "w2": np.asarray(e2.response, float),
        "alive_final": float(np.mean(dust.alive[-1])),
        "T_peak": float(np.nanmax(dust.temperature_K)),
        "mass_ratio": float(diag["accretion_mass_ratio"]),
    }


def interp_echo(pop, trel):
    return (
        np.interp(trel, pop["time"], pop["w1"], left=0.0, right=0.0),
        np.interp(trel, pop["time"], pop["w2"], left=0.0, right=0.0),
    )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    obs = g.load_observations()
    mjd = obs["mjd"].to_numpy(float)
    y1 = obs["F_W1_Jy"].to_numpy(float)
    y2 = obs["F_W2_Jy"].to_numpy(float)
    s1 = obs["sigma_W1_Jy"].to_numpy(float)
    s2 = obs["sigma_W2_Jy"].to_numpy(float)
    b1 = float(obs["baseline_W1_mJy"].iloc[0]) / 1000.0
    b2 = float(obs["baseline_W2_mJy"].iloc[0]) / 1000.0

    pairs = []
    for c in COMPACT_ZONES:
        for e in EXTENDED_ZONES:
            # Strictly non-overlapping radial populations.
            if c[1] <= e[0]:
                for pc in P_VALUES:
                    for pe in P_VALUES:
                        pairs.append((c, pc, e, pe))

    print("=" * 82)
    print("UGC 11487 V4.1 — STRUCTURED TWO-POPULATION PHYSICAL TEST")
    print(f"Radial population pairs : {len(pairs)}")
    print(f"Common t0 values        : {len(T0_GRID)}")
    print(f"Data points             : {2*len(obs)}")
    print(f"Driver                  : {STARS_REL}")
    print(f"Dust                    : {COMPOSITION}, a={GRAIN_CONFIG[0]}..{GRAIN_CONFIG[1]} micron")
    print(f"f_abs cap               : each <= {FABS_COMPONENT_MAX:.2f}, total <= {FABS_TOTAL_MAX:.2f}")
    print("=" * 82)

    # Precompute every unique physical population once.
    unique_pops = sorted({
        (z[0], z[1], p)
        for c, pc, e, pe in pairs
        for z, p in ((c, pc), (e, pe))
    })
    cache = {}
    t_start = time.perf_counter()
    for i, key in enumerate(unique_pops, 1):
        t0 = time.perf_counter()
        cache[key] = build_population_echo(*key)
        q = cache[key]
        print(
            f"[population {i:02d}/{len(unique_pops):02d}] "
            f"R={key[0]:.2f}-{key[1]:.2f} pc p={key[2]:.0f} "
            f"alive={q['alive_final']:.3f} mass={q['mass_ratio']:.12f} "
            f"{time.perf_counter()-t0:.1f}s"
        )

    rows = []
    best_payload = None
    best_chi = np.inf

    for ipair, (cz, pc, ez, pe) in enumerate(pairs, 1):
        cp = cache[(cz[0], cz[1], pc)]
        ep = cache[(ez[0], ez[1], pe)]

        pair_best = None
        for source_t0 in T0_GRID:
            trel = mjd - source_t0
            c1, c2 = interp_echo(cp, trel)
            e1, e2 = interp_echo(ep, trel)

            f, m1, m2, chi1, chi2 = weighted_nonnegative_two_component_fit(
                y1, y2, s1, s2, b1, b2, c1, c2, e1, e2
            )
            chi = chi1 + chi2

            if pair_best is None or chi < pair_best["chi2_total"]:
                pair_best = {
                    "compact_rin_pc": cz[0],
                    "compact_rout_pc": cz[1],
                    "compact_p": pc,
                    "extended_rin_pc": ez[0],
                    "extended_rout_pc": ez[1],
                    "extended_p": pe,
                    "best_t0_mjd": float(source_t0),
                    "f_abs_compact": float(f[0]),
                    "f_abs_extended": float(f[1]),
                    "f_abs_total": float(f.sum()),
                    "chi2_w1": chi1,
                    "chi2_w2": chi2,
                    "chi2_total": chi,
                    "compact_alive_final": cp["alive_final"],
                    "extended_alive_final": ep["alive_final"],
                    "compact_T_peak_K": cp["T_peak"],
                    "extended_T_peak_K": ep["T_peak"],
                    "mass_ratio": cp["mass_ratio"],
                }

            if chi < best_chi:
                best_chi = chi
                best_payload = {
                    "row": dict(pair_best) if pair_best and pair_best["chi2_total"] == chi else {
                        "compact_rin_pc": cz[0], "compact_rout_pc": cz[1], "compact_p": pc,
                        "extended_rin_pc": ez[0], "extended_rout_pc": ez[1], "extended_p": pe,
                        "best_t0_mjd": float(source_t0),
                        "f_abs_compact": float(f[0]), "f_abs_extended": float(f[1]),
                        "f_abs_total": float(f.sum()),
                        "chi2_w1": chi1, "chi2_w2": chi2, "chi2_total": chi,
                        "compact_alive_final": cp["alive_final"],
                        "extended_alive_final": ep["alive_final"],
                        "compact_T_peak_K": cp["T_peak"],
                        "extended_T_peak_K": ep["T_peak"],
                        "mass_ratio": cp["mass_ratio"],
                    },
                    "model1": m1.copy(), "model2": m2.copy()
                }

        rows.append(pair_best)
        print(
            f"[{ipair:03d}/{len(pairs):03d}] "
            f"C={cz[0]:.2f}-{cz[1]:.2f}/p{pc:.0f} "
            f"E={ez[0]:.2f}-{ez[1]:.2f}/p{pe:.0f} "
            f"chi2={pair_best['chi2_total']:.3f} "
            f"(W1={pair_best['chi2_w1']:.1f}, W2={pair_best['chi2_w2']:.1f}) "
            f"f=({pair_best['f_abs_compact']:.3f},{pair_best['f_abs_extended']:.3f}) "
            f"t0={pair_best['best_t0_mjd']:.0f}"
        )

    out = pd.DataFrame(rows).sort_values("chi2_total").reset_index(drop=True)
    out.to_csv(OUTCSV, index=False)

    # Reconstruct global best exactly for output table.
    br = out.iloc[0]
    cp = cache[(float(br.compact_rin_pc), float(br.compact_rout_pc), float(br.compact_p))]
    ep = cache[(float(br.extended_rin_pc), float(br.extended_rout_pc), float(br.extended_p))]
    trel = mjd - float(br.best_t0_mjd)
    c1, c2 = interp_echo(cp, trel)
    e1, e2 = interp_echo(ep, trel)
    f, m1, m2, chi1, chi2 = weighted_nonnegative_two_component_fit(
        y1, y2, s1, s2, b1, b2, c1, c2, e1, e2
    )

    tab = pd.DataFrame({
        "mjd": mjd,
        "W1_data_mJy": 1000*y1,
        "W1_model_mJy": 1000*m1,
        "W1_sigma_mJy": 1000*s1,
        "W1_resid_sigma": (y1-m1)/s1,
        "W2_data_mJy": 1000*y2,
        "W2_model_mJy": 1000*m2,
        "W2_sigma_mJy": 1000*s2,
        "W2_resid_sigma": (y2-m2)/s2,
    })
    tab.to_csv(OUTDIR / "v41_best_epoch_table.csv", index=False)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]}
    )
    ax = axes[0]
    ax.errorbar(mjd, 1000*y1, yerr=1000*s1, fmt="o", label="WISE W1 data")
    ax.errorbar(mjd, 1000*y2, yerr=1000*s2, fmt="s", label="WISE W2 data")
    ax.plot(mjd, 1000*m1, "-o", ms=3, label="V4.1 W1")
    ax.plot(mjd, 1000*m2, "-s", ms=3, label="V4.1 W2")
    ax.set_ylabel("Flux density (mJy)")
    ax.legend()
    ax.set_title(
        f"UGC 11487 V4.1 structured dust | chi2={chi1+chi2:.2f} "
        f"(W1={chi1:.2f}, W2={chi2:.2f})"
    )

    ax = axes[1]
    ax.axhline(0.0, lw=1)
    ax.plot(mjd, (y1-m1)/s1, "-o", label="W1")
    ax.plot(mjd, (y2-m2)/s2, "-s", label="W2")
    ax.set_xlabel("MJD")
    ax.set_ylabel("(data-model)/sigma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "v41_best_residuals.png", dpi=180)
    plt.close(fig)

    manifest = {
        "experiment": "V4.1 minimal structured two-population test",
        "driver": STARS_REL,
        "eta_rad": ETA_RAD,
        "tau_visc_days": TAU_VISC_DAYS,
        "composition": COMPOSITION,
        "grain_config": GRAIN_CONFIG,
        "compact_zones": COMPACT_ZONES,
        "extended_zones": EXTENDED_ZONES,
        "p_values": P_VALUES,
        "t0_grid": [float(x) for x in T0_GRID],
        "f_abs_component_max": FABS_COMPONENT_MAX,
        "f_abs_total_max": FABS_TOTAL_MAX,
        "n_radial_pairs": len(pairs),
        "best": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                 for k, v in br.to_dict().items()},
        "wall_time_s": time.perf_counter() - t_start,
    }
    (OUTDIR / "v41_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 82)
    print("V4.1 STRUCTURED TEST COMPLETE")
    print(f"Successful radial pairs : {len(out)}")
    print(f"BEST chi2_total        : {chi1+chi2:.6f}")
    print(f"BEST chi2_W1           : {chi1:.6f}")
    print(f"BEST chi2_W2           : {chi2:.6f}")
    print(
        f"BEST compact           : {br.compact_rin_pc:.2f}-{br.compact_rout_pc:.2f} pc, "
        f"p={br.compact_p:.0f}, f_abs={f[0]:.4f}"
    )
    print(
        f"BEST extended          : {br.extended_rin_pc:.2f}-{br.extended_rout_pc:.2f} pc, "
        f"p={br.extended_p:.0f}, f_abs={f[1]:.4f}"
    )
    print(f"BEST f_abs total       : {f.sum():.4f}")
    print(f"BEST t0 MJD            : {br.best_t0_mjd:.1f}")
    print(f"Wall time              : {time.perf_counter()-t_start:.1f} s")
    print(f"Results                : {OUTDIR}")
    print("=" * 82)


if __name__ == "__main__":
    main()
