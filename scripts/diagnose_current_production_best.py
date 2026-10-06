"""
UGC 11487 V4 — diagnostic reconstruction of the CURRENT best production block.

This script DOES NOT fit, optimize, modify, or stop the production grid.
It only reads results/v4_grid/production_blocks.csv, selects the current
minimum chi^2 row, reconstructs that exact physical block with the same V4
forward pipeline, and evaluates it at the 22 observed WISE epochs.

Run from project root:
    python scripts\diagnose_current_production_best.py

Outputs:
    results\v4_grid\diagnostics_current_best\
        current_best_lightcurves_residuals.png
        current_best_epoch_table.csv
        current_best_summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import the production engine itself so this diagnostic uses the SAME
# constants, data paths, WISE calibration, geometry, and numerical conventions.
import run_v4_model_grid as grid

from src.v4_driver import build_physical_driver
from src.v4_dust import generate_clumpy_geometry
from src.v4_dust_response import build_dust_response
from src.v4_opacity import load_dust_opacity, make_grain_size_grid
from src.v4_thermal import DustThermalEquilibrium
from src.v4_dust_population import build_population_normalization
from src.v4_3d_echo import integrate_radial_response_over_clumps
from src.v4_wise_emission import (
    load_wise_w1_w2,
    load_calspec_vega,
)

CHECKPOINT = grid.OUTDIR / "production_blocks.csv"
OUTDIR = grid.OUTDIR / "diagnostics_current_best"
OUTDIR.mkdir(parents=True, exist_ok=True)


def select_current_best() -> tuple[pd.Series, int]:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    d = pd.read_csv(CHECKPOINT)
    good = d.loc[
        (d["status"] == "ok") &
        np.isfinite(pd.to_numeric(d["chi2_total"], errors="coerce"))
    ].copy()

    if good.empty:
        raise RuntimeError("No successful finite-chi2 rows in production checkpoint.")

    best = good.loc[pd.to_numeric(good["chi2_total"]).idxmin()]
    return best, len(good)


def reconstruct(best: pd.Series):
    obs = grid.load_observations().sort_values("mjd").reset_index(drop=True)

    stars_file = grid.STARS_ROOT / str(best["stars_rel"]).replace("\\", "/")
    if not stars_file.exists():
        raise FileNotFoundError(f"STARS file not found: {stars_file}")

    # 1. Physical TDE luminosity driver
    driver, driver_diag = build_physical_driver(
    stars_file=stars_file,
    M_BH_Msun=grid.M_BH_MSUN,
    eta_rad=float(best["eta_rad"]),
    tau_visc_days=float(best["tau_visc_days"]),
    t_circ_days=grid.T_CIRC_DAYS,
    dt_days=grid.DRIVER_DT_DAYS,
    pre_peak_days=grid.PRE_PEAK_DAYS,
    post_fallback_days=grid.POST_FALLBACK_DAYS,
    )
    td = driver["t_days"].to_numpy(float)
    L = driver["L_bol_erg_s"].to_numpy(float)

    # 2. Dust microphysics + radial grid
    amin = float(best["a_min_micron"])
    amax = float(best["a_max_micron"])
    q = float(best["q_grain"])
    rin = float(best["r_in_pc"])
    rout = float(best["r_out_pc"])
    p = float(best["p_radial"])

    opacity = grid.load_dust_opacity(grid.DRAINE_DIR, str(best["composition"]))
    thermal = grid.DustThermalEquilibrium(
        opacity,
        source_temperature_K=grid.SOURCE_T_K,
    )
    grains = grid.make_grain_size_grid(amin, amax, grid.N_GRAINS)
    radii, rw = grid.radial_grid_and_weights(rin, rout, p, grid.N_RADII)

    _, _, _, lookup = grid._cached_dust_static(str(best["composition"]), amin, amax)
    dust = grid._build_dust_response_cached(
        td, L, radii, grains, lookup, str(best["composition"]), float(best["density_cm3"])
    )

    # 3. Absolute population normalization at f_abs = 1.
    #    The stored best_f_abs_initial is applied only after the expensive echo,
    #    exactly as in the production scan.
    pop = build_population_normalization(
        thermal_solver=thermal,
        grain_radius_micron=grains,
        clump_radius_pc=radii,
        f_abs_initial=1.0,
        q=q,
        clump_weight=rw,
    )

    # 4. Absolute WISE W1/W2 emission
    w1bp, w2bp = load_wise_w1_w2(grid.WISE_DIR)
    vega_wave, vega_flux = load_calspec_vega(grid.VEGA_FILE)
    dl_m = grid.D_L_MPC * 1.0e6 * grid.PC_M

    pergrain_w1 = grid.absolute_population_band_cube(
        opacity, w1bp, grains, dust.temperature_K, dust.alive, q, grid.Z,
        dl_m, vega_wave, vega_flux, grid.WISE_TGRID,
    )
    pergrain_w2 = grid.absolute_population_band_cube(
        opacity, w2bp, grains, dust.temperature_K, dust.alive, q, grid.Z,
        dl_m, vega_wave, vega_flux, grid.WISE_TGRID,
    )

    radial_w1 = pop.total_grain_number * pergrain_w1
    radial_w2 = pop.total_grain_number * pergrain_w2

    # 5. Same deterministic 3-D geometry as production
    geom = generate_clumpy_geometry(
        n_clumps=grid.N_CLUMPS,
        r_in_pc=rin,
        r_out_pc=rout,
        p=p,
        theta_open_deg=float(best["theta_open_deg"]),
        inclination_deg=float(best["inclination_deg"]),
        seed=grid.GEOMETRY_SEED,
    )
    rc = geom["r_pc"].to_numpy(float)
    zc = geom["z_obs_pc"].to_numpy(float)

    e1 = integrate_radial_response_over_clumps(
        dust_time_rest_days=td,
        radius_grid_pc=radii,
        radial_response=radial_w1,
        clump_radius_pc=rc,
        clump_z_obs_pc=zc,
        redshift=grid.Z,
        chunk_size=256,
        source_time_input=True,
    )
    e2 = integrate_radial_response_over_clumps(
        dust_time_rest_days=td,
        radius_grid_pc=radii,
        radial_response=radial_w2,
        clump_radius_pc=rc,
        clump_z_obs_pc=zc,
        redshift=grid.Z,
        output_time_obs_days=e1.time_obs_days,
        chunk_size=256,
        source_time_input=True,
    )

    # 6. Apply the STORED best cheap-grid point. No re-fitting.
    fabs = float(best["best_f_abs_initial"])
    t0 = float(best["best_t0_mjd"])
    b1 = float(obs["baseline_W1_mJy"].iloc[0]) / 1000.0
    b2 = float(obs["baseline_W2_mJy"].iloc[0]) / 1000.0

    mjd = obs["mjd"].to_numpy(float)
    trel = mjd - t0

    echo1_epoch = np.interp(
        trel, e1.time_obs_days, e1.response, left=0.0, right=0.0
    )
    echo2_epoch = np.interp(
        trel, e2.time_obs_days, e2.response, left=0.0, right=0.0
    )

    model1_jy = b1 + fabs * echo1_epoch
    model2_jy = b2 + fabs * echo2_epoch

    y1 = obs["F_W1_Jy"].to_numpy(float)
    y2 = obs["F_W2_Jy"].to_numpy(float)
    s1 = obs["sigma_W1_Jy"].to_numpy(float)
    s2 = obs["sigma_W2_Jy"].to_numpy(float)

    r1 = (y1 - model1_jy) / s1
    r2 = (y2 - model2_jy) / s2

    chi1 = float(np.sum(r1**2))
    chi2 = float(np.sum(r2**2))
    chit = chi1 + chi2

    # Dense observer-frame curves
    model_mjd = t0 + e1.time_obs_days
    dense1_mjy = 1000.0 * (b1 + fabs * e1.response)
    dense2_mjy = 1000.0 * (b2 + fabs * e2.response)

    tab = pd.DataFrame({
        "epoch_id": obs["epoch_id"] if "epoch_id" in obs.columns else np.arange(len(obs)),
        "mjd": mjd,
        "W1_obs_mJy": 1000.0 * y1,
        "W1_sigma_mJy": 1000.0 * s1,
        "W1_model_mJy": 1000.0 * model1_jy,
        "W1_residual_sigma": r1,
        "W1_chi2_contribution": r1**2,
        "W2_obs_mJy": 1000.0 * y2,
        "W2_sigma_mJy": 1000.0 * s2,
        "W2_model_mJy": 1000.0 * model2_jy,
        "W2_residual_sigma": r2,
        "W2_chi2_contribution": r2**2,
    })

    return obs, tab, model_mjd, dense1_mjy, dense2_mjy, chi1, chi2, chit


def make_figure(tab, model_mjd, dense1, dense2, best):
    fig, axes = plt.subplots(
        2, 1, figsize=(11.5, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    ax, ar = axes

    ax.errorbar(
        tab["mjd"], tab["W1_obs_mJy"], yerr=tab["W1_sigma_mJy"],
        fmt="o", ms=5, capsize=2, label="WISE W1 data"
    )
    ax.errorbar(
        tab["mjd"], tab["W2_obs_mJy"], yerr=tab["W2_sigma_mJy"],
        fmt="s", ms=5, capsize=2, label="WISE W2 data"
    )
    ax.plot(model_mjd, dense1, lw=1.8, label="Current-best V4 W1")
    ax.plot(model_mjd, dense2, lw=1.8, label="Current-best V4 W2")
    ax.set_ylabel("Flux density (mJy)")
    ax.set_xlim(tab["mjd"].min() - 150, tab["mjd"].max() + 150)
    ax.legend(ncol=2)
    ax.set_title(
        f"UGC 11487 V4 current production best | "
        f"chi2 = {float(best['chi2_total']):.2f} | block {best['block_id']}"
    )

    ar.axhline(0.0, lw=1.0)
    ar.plot(tab["mjd"], tab["W1_residual_sigma"], "o-", label="W1")
    ar.plot(tab["mjd"], tab["W2_residual_sigma"], "s-", label="W2")
    ar.set_xlabel("MJD")
    ar.set_ylabel("(data-model)/sigma")
    ar.legend(ncol=2)

    fig.tight_layout()
    out = OUTDIR / "current_best_lightcurves_residuals.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_summary(best, n_good, tab, chi1, chi2, chit):
    # Biggest individual chi2 contributions
    w1_order = np.argsort(tab["W1_chi2_contribution"].to_numpy())[::-1][:5]
    w2_order = np.argsort(tab["W2_chi2_contribution"].to_numpy())[::-1][:5]

    lines = [
        "UGC 11487 V4 — CURRENT PRODUCTION BEST DIAGNOSTIC",
        "=" * 72,
        "NO NEW FITTING OR OPTIMIZATION WAS PERFORMED.",
        "",
        f"successful checkpoint rows = {n_good}",
        f"block_id                   = {best['block_id']}",
        f"STARS                      = {best['stars_rel']}",
        f"eta_rad                    = {float(best['eta_rad']):.6g}",
        f"tau_visc_days              = {float(best['tau_visc_days']):.6g}",
        f"composition                = {best['composition']}",
        f"a_min..a_max micron        = {float(best['a_min_micron']):.4g} .. {float(best['a_max_micron']):.4g}",
        f"q_grain                    = {float(best['q_grain']):.4g}",
        f"Rin..Rout pc               = {float(best['r_in_pc']):.4g} .. {float(best['r_out_pc']):.4g}",
        f"p_radial                   = {float(best['p_radial']):.4g}",
        f"theta_open_deg             = {float(best['theta_open_deg']):.4g}",
        f"inclination_deg            = {float(best['inclination_deg']):.4g}",
        f"best_f_abs_initial         = {float(best['best_f_abs_initial']):.6g}",
        f"best_t0_mjd                = {float(best['best_t0_mjd']):.6f}",
        "",
        "CHECKPOINT CHI-SQUARE:",
        f"chi2_W1 stored             = {float(best['chi2_w1']):.9f}",
        f"chi2_W2 stored             = {float(best['chi2_w2']):.9f}",
        f"chi2_total stored          = {float(best['chi2_total']):.9f}",
        "",
        "RECONSTRUCTED CHI-SQUARE:",
        f"chi2_W1 reconstructed      = {chi1:.9f}",
        f"chi2_W2 reconstructed      = {chi2:.9f}",
        f"chi2_total reconstructed   = {chit:.9f}",
        f"delta total                = {chit - float(best['chi2_total']):+.6e}",
        "",
        "FIVE LARGEST W1 CHI2 CONTRIBUTIONS:",
    ]

    for j in w1_order:
        row = tab.iloc[j]
        lines.append(
            f"MJD {row['mjd']:.3f}: residual={row['W1_residual_sigma']:+.3f} sigma, "
            f"chi2_i={row['W1_chi2_contribution']:.3f}"
        )

    lines += ["", "FIVE LARGEST W2 CHI2 CONTRIBUTIONS:"]
    for j in w2_order:
        row = tab.iloc[j]
        lines.append(
            f"MJD {row['mjd']:.3f}: residual={row['W2_residual_sigma']:+.3f} sigma, "
            f"chi2_i={row['W2_chi2_contribution']:.3f}"
        )

    (OUTDIR / "current_best_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    import argparse
    global CHECKPOINT, OUTDIR
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    args=ap.parse_args()
    CHECKPOINT=args.checkpoint
    OUTDIR=CHECKPOINT.parent / (CHECKPOINT.stem + '_diagnostics')
    OUTDIR.mkdir(parents=True,exist_ok=True)
    best, n_good = select_current_best()

    print("=" * 78)
    print("UGC 11487 V4 — CURRENT PRODUCTION BEST RECONSTRUCTION")
    print("=" * 78)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Successful rows: {n_good}")
    print(f"Current best block: {best['block_id']}")
    print(f"Stored chi2: {float(best['chi2_total']):.6f}")
    print("Reconstructing the exact block. Production is NOT modified.")

    obs, tab, model_mjd, dense1, dense2, chi1, chi2, chit = reconstruct(best)

    if not np.isclose(chit, float(best['chi2_total']), rtol=1e-8, atol=1e-6):
        raise RuntimeError("Reconstruction differs from checkpoint: do not mix code/data versions")
    pd.DataFrame({'mjd':model_mjd,'W1_model_mJy':dense1,'W2_model_mJy':dense2}).to_csv(OUTDIR/'full_lightcurves.csv',index=False)
    tab.to_csv(OUTDIR / "current_best_epoch_table.csv", index=False)
    make_figure(tab, model_mjd, dense1, dense2, best)
    write_summary(best, n_good, tab, chi1, chi2, chit)

    print("")
    print(f"Reconstructed chi2 W1    = {chi1:.6f}")
    print(f"Reconstructed chi2 W2    = {chi2:.6f}")
    print(f"Reconstructed chi2 total = {chit:.6f}")
    print(f"Stored chi2 total        = {float(best['chi2_total']):.6f}")
    print(f"Delta                     = {chit - float(best['chi2_total']):+.6e}")
    print("")
    print(f"Outputs written to: {OUTDIR}")
    print("DONE")


if __name__ == "__main__":
    main()
