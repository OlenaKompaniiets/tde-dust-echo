"""
UGC 11487 V4.2 — two-population composition / grain-size segregation test.

Purpose
-------
Keep the validated fixed-V4 physical core and the V4.1 two-population radial
architecture, but allow compact and extended populations to have independent
composition and grain-size distributions.

No dust temperatures, lags, W1/W2 ratios, centroid offsets, widths, or fluence
ratios are fitted as extra targets.  The likelihood remains the direct 44-point
WISE W1/W2 chi-square.

This is a controlled falsification test of spectral dust physics.  To avoid
simultaneously reopening every dimension:
  * one STARS driver is used (the current control-best driver);
  * eta=0.10 and tau_visc=0 d are fixed;
  * geometry angles remain fixed;
  * radial zones are centered on the best V4.1 solution, with modest variants;
  * compact and extended populations independently choose graphite/silicate
    and one of three grain-size ranges;
  * p is independently varied in each population;
  * the two physical absorbed fractions are solved non-negatively at each
    common t0, with f_comp + f_ext <= 0.8.

Run:
    python scripts\\run_v42_composition_size_test.py

Outputs:
    results\\v42_composition_size_test\\v42_results.csv
    results\\v42_composition_size_test\\v42_best_epoch_table.csv
    results\\v42_composition_size_test\\v42_best_residuals.png
    results\\v42_composition_size_test\\v42_manifest.json
"""

from __future__ import annotations

import json
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

import scripts.run_v4_model_grid as g

OUTDIR = PROJECT_ROOT / "results" / "v42_composition_size_test"
OUTCSV = OUTDIR / "v42_results.csv"

# ---------------------------------------------------------------------------
# Fixed driver / geometry for this controlled spectral-physics experiment.
# ---------------------------------------------------------------------------
STARS_REL = "input/m0.3_t0.0/0.900.dat"
ETA_RAD = 0.10
TAU_VISC_DAYS = 0.0
DENSITY_CM3 = 1.0e9
THETA_OPEN_DEG = 60.0
INCLINATION_DEG = 55.0

# V4.1 best was C=0.15-0.40 pc, E=0.50-3.00 pc, pC=pE=0.
# Keep that solution and nearby radial alternatives, rather than reopening the
# entire radial grid while testing composition / size segregation.
COMPACT_ZONES = (
    (0.10, 0.30),
    (0.15, 0.40),
    (0.15, 0.50),
)
EXTENDED_ZONES = (
    (0.40, 1.50),
    (0.50, 3.00),
    (0.75, 3.00),
)
P_VALUES = (0.0, 1.0, 2.0)

COMPOSITIONS = ("graphite", "silicate")

# (amin micron, amax micron, q)
# Includes the V4.1 distribution plus small-grain- and large-grain-weighted
# alternatives without introducing a free q in this first V4.2 test.
GRAIN_CONFIGS = (
    (0.010, 0.25, 3.5),
    (0.010, 1.00, 3.5),
    (0.100, 1.00, 3.5),
)

T0_GRID = np.arange(56000.0, 58000.1, 20.0)

FABS_COMPONENT_MAX = 0.80
FABS_TOTAL_MAX = 0.80


def solve_two_amplitudes(y1, y2, s1, s2, base1, base2,
                         c1, c2, e1, e2):
    """
    Weighted non-negative two-population fit.

    One coefficient per physical dust population scales W1 and W2 together.
    There are no independent band amplitudes.
    """
    y = np.concatenate(((y1-base1)/s1, (y2-base2)/s2))
    A = np.column_stack((
        np.concatenate((c1/s1, c2/s2)),
        np.concatenate((e1/s1, e2/s2)),
    ))

    sol = lsq_linear(
        A, y,
        bounds=(np.zeros(2), np.full(2, FABS_COMPONENT_MAX)),
        method="trf", lsmr_tol="auto"
    )
    f = np.asarray(sol.x, float)

    if f.sum() > FABS_TOTAL_MAX:
        d = A[:, 0] - A[:, 1]
        y0 = y - A[:, 1]*FABS_TOTAL_MAX
        den = float(np.dot(d, d))
        fc = 0.5*FABS_TOTAL_MAX if den <= 0 else float(np.dot(d, y0)/den)
        fc = float(np.clip(fc, 0.0, FABS_TOTAL_MAX))
        f = np.array([fc, FABS_TOTAL_MAX-fc], float)

    m1 = base1 + f[0]*c1 + f[1]*e1
    m2 = base2 + f[0]*c2 + f[1]*e2
    r1 = (y1-m1)/s1
    r2 = (y2-m2)/s2
    return f, m1, m2, float(np.sum(r1*r1)), float(np.sum(r2*r2))


@lru_cache(maxsize=None)
def build_population_echo(rin, rout, p, composition, amin, amax, q):
    """f_abs=1 physical W1/W2 observer-frame echo for one dust population."""
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

    opacity, thermal, grains, lookup = g._cached_dust_static(
        composition, float(amin), float(amax)
    )
    radii, rw = g.radial_grid_and_weights(
        float(rin), float(rout), float(p), g.N_RADII
    )

    dust = g._build_dust_response_cached(
        time_days=td,
        luminosity_erg_s=L,
        radius_pc=radii,
        grains=grains,
        lookup=lookup,
        composition=composition,
        density_cm3=DENSITY_CM3,
    )

    pop = g.build_population_normalization(
        thermal_solver=thermal,
        grain_radius_micron=grains,
        clump_radius_pc=radii,
        f_abs_initial=1.0,
        q=float(q),
        clump_weight=rw,
    )

    w1, w2, vw, vf = g._cached_wise_static()
    dl_m = g.D_L_MPC * 1.0e6 * g.PC_M

    pg1 = g.absolute_population_band_cube(
        opacity, w1, grains, dust.temperature_K, dust.alive, float(q), g.Z,
        dl_m, vw, vf, g.WISE_TGRID,
    )
    pg2 = g.absolute_population_band_cube(
        opacity, w2, grains, dust.temperature_K, dust.alive, float(q), g.Z,
        dl_m, vw, vf, g.WISE_TGRID,
    )
    rw1 = pop.total_grain_number * pg1
    rw2 = pop.total_grain_number * pg2

    rc, zc = g._cached_geometry(
        float(rin), float(rout), float(p),
        THETA_OPEN_DEG, INCLINATION_DEG
    )
    e1, e2 = g.integrate_radial_responses_over_clumps(
        dust_time_rest_days=td,
        radius_grid_pc=radii,
        radial_responses=np.stack((rw1, rw2), axis=0),
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


def interp(pop, trel):
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
    b1 = float(obs["baseline_W1_mJy"].iloc[0])/1000.0
    b2 = float(obs["baseline_W2_mJy"].iloc[0])/1000.0

    # Every population state to be physically calculated once.
    pop_keys = []
    for zone_set in (COMPACT_ZONES, EXTENDED_ZONES):
        for rin, rout in zone_set:
            for p in P_VALUES:
                for comp in COMPOSITIONS:
                    for amin, amax, q in GRAIN_CONFIGS:
                        pop_keys.append((rin, rout, p, comp, amin, amax, q))
    pop_keys = list(dict.fromkeys(pop_keys))

    print("="*88)
    print("UGC 11487 V4.2 — COMPOSITION / GRAIN-SIZE SEGREGATION TEST")
    print(f"Unique physical populations : {len(pop_keys)}")
    print(f"Common t0 values            : {len(T0_GRID)}")
    print(f"Data points                 : {2*len(obs)}")
    print(f"Driver                      : {STARS_REL}")
    print(f"f_abs cap                   : each <= {FABS_COMPONENT_MAX:.2f}, total <= {FABS_TOTAL_MAX:.2f}")
    print("="*88)

    cache = {}
    tstart = time.perf_counter()
    for i, key in enumerate(pop_keys, 1):
        t = time.perf_counter()
        cache[key] = build_population_echo(*key)
        qd = cache[key]
        print(
            f"[population {i:03d}/{len(pop_keys):03d}] "
            f"R={key[0]:.2f}-{key[1]:.2f} p={key[2]:.0f} "
            f"{key[3]:8s} a={key[4]:.3f}-{key[5]:.2f} "
            f"alive={qd['alive_final']:.3f} mass={qd['mass_ratio']:.12f} "
            f"{time.perf_counter()-t:.1f}s"
        )

    # Search pair combinations. This is cheap after physical echoes are cached.
    rows = []
    n_pairs = (
        len(COMPACT_ZONES)*len(EXTENDED_ZONES)*
        len(P_VALUES)**2*len(COMPOSITIONS)**2*len(GRAIN_CONFIGS)**2
    )
    ipair = 0
    global_best = None

    for cz in COMPACT_ZONES:
        for ez in EXTENDED_ZONES:
            if cz[1] > ez[0]:
                continue
            for pc in P_VALUES:
                for pe in P_VALUES:
                    for cc in COMPOSITIONS:
                        for ce in COMPOSITIONS:
                            for gc in GRAIN_CONFIGS:
                                for ge in GRAIN_CONFIGS:
                                    ipair += 1
                                    ck = (cz[0], cz[1], pc, cc, *gc)
                                    ek = (ez[0], ez[1], pe, ce, *ge)
                                    cp, ep = cache[ck], cache[ek]

                                    pair_best = None
                                    for source_t0 in T0_GRID:
                                        trel = mjd-source_t0
                                        c1, c2 = interp(cp, trel)
                                        e1, e2 = interp(ep, trel)
                                        f, m1, m2, chi1, chi2 = solve_two_amplitudes(
                                            y1, y2, s1, s2, b1, b2,
                                            c1, c2, e1, e2
                                        )
                                        chi = chi1+chi2
                                        if pair_best is None or chi < pair_best["chi2_total"]:
                                            pair_best = {
                                                "compact_rin_pc": cz[0],
                                                "compact_rout_pc": cz[1],
                                                "compact_p": pc,
                                                "compact_composition": cc,
                                                "compact_amin_um": gc[0],
                                                "compact_amax_um": gc[1],
                                                "compact_q": gc[2],
                                                "extended_rin_pc": ez[0],
                                                "extended_rout_pc": ez[1],
                                                "extended_p": pe,
                                                "extended_composition": ce,
                                                "extended_amin_um": ge[0],
                                                "extended_amax_um": ge[1],
                                                "extended_q": ge[2],
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

                                    rows.append(pair_best)
                                    if global_best is None or pair_best["chi2_total"] < global_best["chi2_total"]:
                                        global_best = dict(pair_best)

                                    # Keep console readable: report new global minima and every 250 pairs.
                                    if (len(rows) == 1 or
                                        pair_best["chi2_total"] <= global_best["chi2_total"] + 1e-12 or
                                        ipair % 250 == 0):
                                        print(
                                            f"[pair {ipair:04d}] best={pair_best['chi2_total']:.3f} "
                                            f"global={global_best['chi2_total']:.3f} "
                                            f"C:{cc[:3]} {gc[0]:.2f}-{gc[1]:.2f} "
                                            f"E:{ce[:3]} {ge[0]:.2f}-{ge[1]:.2f}"
                                        )

    out = pd.DataFrame(rows).sort_values("chi2_total").reset_index(drop=True)
    out.to_csv(OUTCSV, index=False)

    br = out.iloc[0]
    ck = (
        float(br.compact_rin_pc), float(br.compact_rout_pc), float(br.compact_p),
        str(br.compact_composition), float(br.compact_amin_um),
        float(br.compact_amax_um), float(br.compact_q)
    )
    ek = (
        float(br.extended_rin_pc), float(br.extended_rout_pc), float(br.extended_p),
        str(br.extended_composition), float(br.extended_amin_um),
        float(br.extended_amax_um), float(br.extended_q)
    )
    cp, ep = cache[ck], cache[ek]
    trel = mjd-float(br.best_t0_mjd)
    c1, c2 = interp(cp, trel)
    e1, e2 = interp(ep, trel)
    f, m1, m2, chi1, chi2 = solve_two_amplitudes(
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
    tab.to_csv(OUTDIR/"v42_best_epoch_table.csv", index=False)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]}
    )
    ax = axes[0]
    ax.errorbar(mjd, 1000*y1, yerr=1000*s1, fmt="o", label="WISE W1 data")
    ax.errorbar(mjd, 1000*y2, yerr=1000*s2, fmt="s", label="WISE W2 data")
    ax.plot(mjd, 1000*m1, "-o", ms=3, label="V4.2 W1")
    ax.plot(mjd, 1000*m2, "-s", ms=3, label="V4.2 W2")
    ax.set_ylabel("Flux density (mJy)")
    ax.legend()
    ax.set_title(
        f"UGC 11487 V4.2 composition/size segregation | "
        f"chi2={chi1+chi2:.2f} (W1={chi1:.2f}, W2={chi2:.2f})"
    )

    ax = axes[1]
    ax.axhline(0.0, lw=1)
    ax.plot(mjd, (y1-m1)/s1, "-o", label="W1")
    ax.plot(mjd, (y2-m2)/s2, "-s", label="W2")
    ax.set_xlabel("MJD")
    ax.set_ylabel("(data-model)/sigma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR/"v42_best_residuals.png", dpi=180)
    plt.close(fig)

    manifest = {
        "experiment": "V4.2 composition/grain-size segregation test",
        "driver": STARS_REL,
        "eta_rad": ETA_RAD,
        "tau_visc_days": TAU_VISC_DAYS,
        "density_cm3": DENSITY_CM3,
        "theta_open_deg": THETA_OPEN_DEG,
        "inclination_deg": INCLINATION_DEG,
        "compact_zones": COMPACT_ZONES,
        "extended_zones": EXTENDED_ZONES,
        "p_values": P_VALUES,
        "compositions": COMPOSITIONS,
        "grain_configs": GRAIN_CONFIGS,
        "t0_grid": [float(x) for x in T0_GRID],
        "f_abs_component_max": FABS_COMPONENT_MAX,
        "f_abs_total_max": FABS_TOTAL_MAX,
        "n_unique_physical_populations": len(pop_keys),
        "n_evaluated_pairs": len(out),
        "best": {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in br.to_dict().items()
        },
        "wall_time_s": time.perf_counter()-tstart,
    }
    (OUTDIR/"v42_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("="*88)
    print("V4.2 TEST COMPLETE")
    print(f"Evaluated pairs        : {len(out)}")
    print(f"BEST chi2_total       : {chi1+chi2:.6f}")
    print(f"BEST chi2_W1          : {chi1:.6f}")
    print(f"BEST chi2_W2          : {chi2:.6f}")
    print(
        f"COMPACT               : {br.compact_rin_pc:.2f}-{br.compact_rout_pc:.2f} pc "
        f"p={br.compact_p:.0f} {br.compact_composition} "
        f"a={br.compact_amin_um:.3f}-{br.compact_amax_um:.2f} um "
        f"f_abs={f[0]:.4f}"
    )
    print(
        f"EXTENDED              : {br.extended_rin_pc:.2f}-{br.extended_rout_pc:.2f} pc "
        f"p={br.extended_p:.0f} {br.extended_composition} "
        f"a={br.extended_amin_um:.3f}-{br.extended_amax_um:.2f} um "
        f"f_abs={f[1]:.4f}"
    )
    print(f"TOTAL f_abs           : {f.sum():.4f}")
    print(f"BEST t0 MJD           : {br.best_t0_mjd:.1f}")
    print(f"Wall time             : {time.perf_counter()-tstart:.1f} s")
    print(f"Results               : {OUTDIR}")
    print("="*88)


if __name__ == "__main__":
    main()
