"""
UGC 11487 V4.3 — temporal-driver test on the validated V4.2 dust architecture.

Question
--------
After V4.2 reduced chi2 from ~1004 to ~524 by allowing structured radial
dust plus composition/grain-size segregation, can physically different STARS
fallback histories and causal viscous smoothing reproduce the remaining
rise/peak morphology?

Kept fixed on purpose:
  compact dust  : 0.15--0.50 pc, p=0, large grains 0.1--1.0 um
  extended dust : 0.50--3.00 pc, p=0, large grains 0.1--1.0 um
  geometry      : theta_open=60 deg, inclination=55 deg
  density       : 1e9 cm^-3
  eta_rad       : 0.10

Two dust-composition architectures are tested because V4.2 showed near
degeneracy:
  A: compact silicate + extended graphite  (V4.2 global best)
  B: compact graphite + extended graphite  (near-degenerate control)

For every STARS curve and tau_visc in
  0, 25, 50, 100, 200, 300, 500 d
the code computes the full physical dust response.  It first scans t0 on a
20-d grid and then refines t0 continuously around the coarse minimum.
At each t0 the two physical f_abs coefficients are solved non-negatively with
f_comp + f_ext <= 0.8.

No empirical lags, temperatures, widths, fluence ratios, or W1/W2-specific
amplitudes enter the objective.  The likelihood is still the direct 44-point
WISE W1/W2 chi-square.

IMPORTANT:
This script does NOT apply a STARS rp/rg validity cut because the exact stellar
radii are not yet in the project.  Results are therefore numerical/physical
model tests; final astrophysical interpretation must later flag invalid STARS
curves using the official rp > 10 rg criterion.

Run:
    python scripts\\run_v43_temporal_driver_test.py --workers 16

Resume:
    run the same command again; completed blocks are skipped.

Outputs:
    results\\v43_temporal_driver_test\\v43_blocks.csv
    results\\v43_temporal_driver_test\\v43_best_epoch_table.csv
    results\\v43_temporal_driver_test\\v43_best_residuals.png
    results\\v43_temporal_driver_test\\v43_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear, minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_v4_model_grid as g

OUTDIR = PROJECT_ROOT / "results" / "v43_temporal_driver_test"
OUTCSV = OUTDIR / "v43_blocks.csv"
ERRORS = OUTDIR / "v43_errors.jsonl"

ETA_RAD = 0.10
TAUS = (0.0, 25.0, 50.0, 100.0, 200.0, 300.0, 500.0)
DENSITY_CM3 = 1.0e9
THETA_OPEN_DEG = 60.0
INCLINATION_DEG = 55.0
GRAIN = (0.10, 1.00, 3.5)

COMPACT = (0.15, 0.50, 0.0)   # Rin, Rout, p
EXTENDED = (0.50, 3.00, 0.0)

ARCHITECTURES = (
    ("silicate_graphite", "silicate", "graphite"),
    ("graphite_graphite", "graphite", "graphite"),
)

T0_COARSE = np.arange(56000.0, 58000.1, 20.0)
T0_REFINE_HALF_WIDTH = 25.0
FABS_COMPONENT_MAX = 0.80
FABS_TOTAL_MAX = 0.80


def solve_amp(y1, y2, s1, s2, b1, b2, c1, c2, e1, e2):
    y = np.concatenate(((y1-b1)/s1, (y2-b2)/s2))
    A = np.column_stack((
        np.concatenate((c1/s1, c2/s2)),
        np.concatenate((e1/s1, e2/s2)),
    ))
    sol = lsq_linear(
        A, y, bounds=(np.zeros(2), np.full(2, FABS_COMPONENT_MAX)),
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

    m1 = b1 + f[0]*c1 + f[1]*e1
    m2 = b2 + f[0]*c2 + f[1]*e2
    r1 = (y1-m1)/s1
    r2 = (y2-m2)/s2
    return f, m1, m2, float(np.dot(r1,r1)), float(np.dot(r2,r2))


def build_echo(stars_rel, tau, zone, composition):
    stars_file = g.STARS_ROOT / stars_rel
    driver, diag = g.build_physical_driver(
        stars_file=stars_file,
        M_BH_Msun=g.M_BH_MSUN,
        eta_rad=ETA_RAD,
        tau_visc_days=float(tau),
        t_circ_days=g.T_CIRC_DAYS,
        dt_days=g.DRIVER_DT_DAYS,
        pre_peak_days=g.PRE_PEAK_DAYS,
        post_fallback_days=g.POST_FALLBACK_DAYS,
    )
    td = driver["t_days"].to_numpy(float)
    L = driver["L_bol_erg_s"].to_numpy(float)

    rin, rout, p = zone
    amin, amax, q = GRAIN
    opacity, thermal, grains, lookup = g._cached_dust_static(
        composition, amin, amax
    )
    radii, rw = g.radial_grid_and_weights(rin, rout, p, g.N_RADII)
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
        q=q,
        clump_weight=rw,
    )

    w1, w2, vw, vf = g._cached_wise_static()
    dl_m = g.D_L_MPC*1e6*g.PC_M
    pg1 = g.absolute_population_band_cube(
        opacity, w1, grains, dust.temperature_K, dust.alive, q, g.Z,
        dl_m, vw, vf, g.WISE_TGRID
    )
    pg2 = g.absolute_population_band_cube(
        opacity, w2, grains, dust.temperature_K, dust.alive, q, g.Z,
        dl_m, vw, vf, g.WISE_TGRID
    )
    radial1 = pop.total_grain_number*pg1
    radial2 = pop.total_grain_number*pg2

    rc, zc = g._cached_geometry(
        rin, rout, p, THETA_OPEN_DEG, INCLINATION_DEG
    )
    e1, e2 = g.integrate_radial_responses_over_clumps(
        dust_time_rest_days=td,
        radius_grid_pc=radii,
        radial_responses=np.stack((radial1, radial2), axis=0),
        clump_radius_pc=rc,
        clump_z_obs_pc=zc,
        redshift=g.Z,
        chunk_size=256,
        source_time_input=True,
    )
    return {
        "time": np.asarray(e1.time_obs_days,float),
        "w1": np.asarray(e1.response,float),
        "w2": np.asarray(e2.response,float),
        "alive": float(np.mean(dust.alive[-1])),
        "Tpeak": float(np.nanmax(dust.temperature_K)),
        "mass_ratio": float(diag["accretion_mass_ratio"]),
        "Lpeak": float(diag.get("L_peak_erg_s", np.nan)),
    }


def interp(pop, trel):
    return (
        np.interp(trel, pop["time"], pop["w1"], left=0.0, right=0.0),
        np.interp(trel, pop["time"], pop["w2"], left=0.0, right=0.0),
    )


def evaluate_one(task):
    stars_rel, tau, arch_name, comp_c, comp_e, obsrec = task
    obs = pd.DataFrame(obsrec)
    mjd = obs.mjd.to_numpy(float)
    y1 = obs.F_W1_Jy.to_numpy(float)
    y2 = obs.F_W2_Jy.to_numpy(float)
    s1 = obs.sigma_W1_Jy.to_numpy(float)
    s2 = obs.sigma_W2_Jy.to_numpy(float)
    b1 = float(obs.baseline_W1_mJy.iloc[0])/1000.0
    b2 = float(obs.baseline_W2_mJy.iloc[0])/1000.0

    cp = build_echo(stars_rel, tau, COMPACT, comp_c)
    ep = build_echo(stars_rel, tau, EXTENDED, comp_e)

    def at_t0(t0, payload=False):
        trel = mjd-float(t0)
        c1,c2 = interp(cp,trel)
        e1,e2 = interp(ep,trel)
        f,m1,m2,ch1,ch2 = solve_amp(y1,y2,s1,s2,b1,b2,c1,c2,e1,e2)
        if payload:
            return ch1+ch2, f, m1, m2, ch1, ch2
        return ch1+ch2

    coarse_vals = np.array([at_t0(t) for t in T0_COARSE])
    j = int(np.argmin(coarse_vals))
    tc = float(T0_COARSE[j])
    lo = max(float(T0_COARSE[0]), tc-T0_REFINE_HALF_WIDTH)
    hi = min(float(T0_COARSE[-1]), tc+T0_REFINE_HALF_WIDTH)
    opt = minimize_scalar(
        at_t0, bounds=(lo,hi), method="bounded",
        options={"xatol":0.05, "maxiter":80}
    )
    tbest = float(opt.x)
    chi,f,m1,m2,ch1,ch2 = at_t0(tbest,payload=True)

    bid = hashlib.sha1(
        f"{stars_rel}|{tau}|{arch_name}".encode()
    ).hexdigest()[:16]
    return {
        "block_id":bid,
        "stars_rel":stars_rel,
        "tau_visc_days":float(tau),
        "architecture":arch_name,
        "compact_composition":comp_c,
        "extended_composition":comp_e,
        "best_t0_mjd":tbest,
        "f_abs_compact":float(f[0]),
        "f_abs_extended":float(f[1]),
        "f_abs_total":float(f.sum()),
        "chi2_w1":ch1,
        "chi2_w2":ch2,
        "chi2_total":chi,
        "compact_alive_final":cp["alive"],
        "extended_alive_final":ep["alive"],
        "compact_Tpeak_K":cp["Tpeak"],
        "extended_Tpeak_K":ep["Tpeak"],
        "mass_ratio":cp["mass_ratio"],
    }


def discover_stars():
    files = sorted(g.STARS_ROOT.glob("input/**/*.dat"))
    if not files:
        raise RuntimeError(f"No STARS .dat files found below {g.STARS_ROOT/'input'}")
    return [p.relative_to(g.STARS_ROOT).as_posix() for p in files]


def completed_ids():
    if not OUTCSV.exists():
        return set()
    try:
        d = pd.read_csv(OUTCSV)
        return set(d.block_id.astype(str))
    except Exception:
        return set()


def append_row(row):
    pd.DataFrame([row]).to_csv(
        OUTCSV, mode="a", header=not OUTCSV.exists(), index=False
    )


def write_best_products(obs, best):
    # Re-evaluate only the best block to obtain epoch models.
    task = (
        best["stars_rel"], float(best["tau_visc_days"]),
        best["architecture"], best["compact_composition"],
        best["extended_composition"], obs.to_dict("records")
    )
    # Build directly here to retain model arrays.
    mjd=obs.mjd.to_numpy(float)
    y1=obs.F_W1_Jy.to_numpy(float); y2=obs.F_W2_Jy.to_numpy(float)
    s1=obs.sigma_W1_Jy.to_numpy(float); s2=obs.sigma_W2_Jy.to_numpy(float)
    b1=float(obs.baseline_W1_mJy.iloc[0])/1000.; b2=float(obs.baseline_W2_mJy.iloc[0])/1000.
    cp=build_echo(best["stars_rel"],float(best["tau_visc_days"]),COMPACT,best["compact_composition"])
    ep=build_echo(best["stars_rel"],float(best["tau_visc_days"]),EXTENDED,best["extended_composition"])
    trel=mjd-float(best["best_t0_mjd"])
    c1,c2=interp(cp,trel); e1,e2=interp(ep,trel)
    f,m1,m2,ch1,ch2=solve_amp(y1,y2,s1,s2,b1,b2,c1,c2,e1,e2)

    tab=pd.DataFrame({
        "mjd":mjd,
        "W1_data_mJy":1000*y1,"W1_model_mJy":1000*m1,
        "W1_sigma_mJy":1000*s1,"W1_resid_sigma":(y1-m1)/s1,
        "W2_data_mJy":1000*y2,"W2_model_mJy":1000*m2,
        "W2_sigma_mJy":1000*s2,"W2_resid_sigma":(y2-m2)/s2,
    })
    tab.to_csv(OUTDIR/"v43_best_epoch_table.csv",index=False)

    fig,axes=plt.subplots(2,1,figsize=(14,10),sharex=True,
        gridspec_kw={"height_ratios":[2.1,1]})
    ax=axes[0]
    ax.errorbar(mjd,1000*y1,yerr=1000*s1,fmt="o",label="WISE W1 data")
    ax.errorbar(mjd,1000*y2,yerr=1000*s2,fmt="s",label="WISE W2 data")
    ax.plot(mjd,1000*m1,"-o",ms=3,label="V4.3 W1")
    ax.plot(mjd,1000*m2,"-s",ms=3,label="V4.3 W2")
    ax.set_ylabel("Flux density (mJy)"); ax.legend()
    ax.set_title(
        f"UGC 11487 V4.3 temporal-driver test | chi2={ch1+ch2:.2f} "
        f"(W1={ch1:.2f}, W2={ch2:.2f})"
    )
    ax=axes[1]; ax.axhline(0,lw=1)
    ax.plot(mjd,(y1-m1)/s1,"-o",label="W1")
    ax.plot(mjd,(y2-m2)/s2,"-s",label="W2")
    ax.set_xlabel("MJD"); ax.set_ylabel("(data-model)/sigma"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR/"v43_best_residuals.png",dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=16)
    args=ap.parse_args()

    OUTDIR.mkdir(parents=True,exist_ok=True)
    obs=g.load_observations()
    stars=discover_stars()
    done=completed_ids()

    tasks=[]
    obsrec=obs.to_dict("records")
    for s in stars:
        for tau in TAUS:
            for an,cc,ce in ARCHITECTURES:
                bid=hashlib.sha1(f"{s}|{tau}|{an}".encode()).hexdigest()[:16]
                if bid not in done:
                    tasks.append((s,tau,an,cc,ce,obsrec))

    total_all=len(stars)*len(TAUS)*len(ARCHITECTURES)
    print("="*92)
    print("UGC 11487 V4.3 — TEMPORAL DRIVER TEST")
    print(f"STARS curves          : {len(stars)}")
    print(f"tau values            : {len(TAUS)} {TAUS}")
    print(f"dust architectures    : {len(ARCHITECTURES)}")
    print(f"total blocks          : {total_all}")
    print(f"already completed     : {len(done)}")
    print(f"remaining             : {len(tasks)}")
    print(f"workers               : {args.workers}")
    print("NOTE: STARS rp>10rg validity gate is NOT yet applied.")
    print("="*92)

    tstart=time.perf_counter()
    errors=0
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs={ex.submit(evaluate_one,t):t for t in tasks}
            for i,fut in enumerate(as_completed(futs),1):
                task=futs[fut]
                try:
                    row=fut.result()
                    append_row(row)
                    print(
                        f"[{i:04d}/{len(tasks):04d}] ok "
                        f"{row['stars_rel']} tau={row['tau_visc_days']:.0f} "
                        f"{row['architecture']} chi2={row['chi2_total']:.3f} "
                        f"t0={row['best_t0_mjd']:.1f} "
                        f"f=({row['f_abs_compact']:.3f},{row['f_abs_extended']:.3f}) "
                        f"mass={row['mass_ratio']:.12f}"
                    )
                except Exception as e:
                    errors+=1
                    err={"task":task[:5],"error":repr(e)}
                    with ERRORS.open("a",encoding="utf-8") as fh:
                        fh.write(json.dumps(err)+"\n")
                    print(f"[{i:04d}/{len(tasks):04d}] ERROR {task[:5]} {e!r}")

    out=pd.read_csv(OUTCSV).sort_values("chi2_total").reset_index(drop=True)
    best=out.iloc[0].to_dict()
    write_best_products(obs,best)

    manifest={
        "experiment":"V4.3 temporal-driver test",
        "n_stars_curves":len(stars),
        "tau_visc_days":TAUS,
        "architectures":ARCHITECTURES,
        "eta_rad":ETA_RAD,
        "compact_zone":COMPACT,
        "extended_zone":EXTENDED,
        "grain_config":GRAIN,
        "density_cm3":DENSITY_CM3,
        "theta_open_deg":THETA_OPEN_DEG,
        "inclination_deg":INCLINATION_DEG,
        "t0_coarse_grid":[float(x) for x in T0_COARSE],
        "t0_refine_half_width_days":T0_REFINE_HALF_WIDTH,
        "f_abs_component_max":FABS_COMPONENT_MAX,
        "f_abs_total_max":FABS_TOTAL_MAX,
        "stars_validity_gate_applied":False,
        "stars_validity_note":"Final interpretation requires official STARS rp > 10 rg scaling validity audit using exact stellar radii.",
        "best":best,
        "wall_time_s":time.perf_counter()-tstart,
        "errors_this_run":errors,
    }
    (OUTDIR/"v43_manifest.json").write_text(
        json.dumps(manifest,indent=2,default=float),encoding="utf-8"
    )

    print("="*92)
    print("V4.3 TEST COMPLETE")
    print(f"successful blocks      : {len(out)} / {total_all}")
    print(f"errors this run         : {errors}")
    print(f"BEST chi2_total        : {best['chi2_total']:.6f}")
    print(f"BEST chi2_W1           : {best['chi2_w1']:.6f}")
    print(f"BEST chi2_W2           : {best['chi2_w2']:.6f}")
    print(f"BEST STARS             : {best['stars_rel']}")
    print(f"BEST tau_visc          : {best['tau_visc_days']:.1f} d")
    print(f"BEST architecture      : {best['architecture']}")
    print(f"BEST t0                : {best['best_t0_mjd']:.2f}")
    print(
        f"BEST f_abs             : ({best['f_abs_compact']:.4f}, "
        f"{best['f_abs_extended']:.4f}) total={best['f_abs_total']:.4f}"
    )
    print(f"BEST mass ratio        : {best['mass_ratio']:.12f}")
    print(f"wall time              : {time.perf_counter()-tstart:.1f} s")
    print(f"results                : {OUTDIR}")
    print("CAUTION: apply STARS rp>10rg validity audit before astrophysical interpretation.")
    print("="*92)


if __name__=="__main__":
    main()
