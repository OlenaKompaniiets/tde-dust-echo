"""
UGC 11487 V4 — production physical model-grid engine (shared-transfer optimized).

Purpose
-------
Evaluate a physically explicit dust-echo grid against the FINAL 22-epoch
WISE data.  This is not the old V3 phenomenological fit.

Forward chain for every expensive physical block:
    STARS fallback -> Lbol(t) -> retarded heating -> radiative equilibrium
    -> irreversible sublimation -> absolute single-grain WISE emission
    -> physically normalized grain population -> 3-D observer echo.

Only after that are the cheap nuisance grids scanned:
    t0_MJD and f_abs,0.

Important
---------
* W1/W2 fluxes and FINAL effective uncertainties are read directly from the
  canonical 22-epoch CSV.  No magnitude reconversion and no new error floor.
* The supplied W1/W2 baselines are fixed observational inputs, not fitted.
* There is no independent W1/W2 amplitude scale.
* f_abs,0 is one common physical absorbed-fraction normalization.  Because
  the present optically-thin/non-overlap population model is linear in grain
  number, the expensive dust solution is calculated at f_abs,0=1 and then
  scaled by the requested f_abs grid.  Temperatures/sublimation are unchanged.
* t0 is a pure translation of an already-computed observer-frame light curve.
* t_circ is kept at zero while t0 is scanned because a pure driver time shift
  is exactly degenerate with t0 in these data.
* Checkpoint rows are appended continuously.  Re-running skips completed
  physical block IDs.

Modes
-----
  python scripts\\run_v4_model_grid.py --mode count
  python scripts\\run_v4_model_grid.py --mode benchmark --max-blocks 8
  python scripts\\run_v4_model_grid.py --mode production --workers 16

The grid below is deliberately explicit.  Edit GRID only when changing the
scientific experiment; the evaluator itself should not be edited to chase a
better fit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

# Prevent BLAS oversubscription before importing numpy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v4_driver import build_physical_driver
from src.v4_dust import generate_clumpy_geometry
from src.v4_dust_response import DustResponseResult, retarded_luminosity
from src.v4_thermal_lookup import build_thermal_lookup, incident_flux_W_m2, lookup_temperature_from_flux
from src.v4_sublimation import sublimation_temperature_K
from src.v4_opacity import load_dust_opacity, make_grain_size_grid
from src.v4_thermal import DustThermalEquilibrium
from src.v4_dust_population import build_population_normalization
from src.v4_3d_echo import (
    integrate_radial_response_over_clumps,
    integrate_radial_responses_over_clumps,
    response_centroid_days,
    response_peak_time_days,
)
from src.v4_wise_emission import (
    load_wise_w1_w2,
    load_calspec_vega,
    fast_population_band_cube,
)

PC_M = 3.085677581491367e16
DAY_S = 86400.0

# -----------------------------------------------------------------------------
# Fixed source / data configuration
# -----------------------------------------------------------------------------
DATA_CSV = PROJECT_ROOT / "data" / "UGC11487_WISE_TDE_model_input_22epochs.csv"
STARS_ROOT = Path(os.environ.get("UGC_STARS_ROOT", PROJECT_ROOT / "STARS_library-master"))
DRAINE_DIR = PROJECT_ROOT / "data" / "reference" / "draine"
WISE_DIR = PROJECT_ROOT / "data" / "reference" / "wise"
VEGA_FILE = PROJECT_ROOT / "data" / "reference" / "calspec" / "alpha_lyr_stis_011.fits"
OUTDIR = PROJECT_ROOT / "results" / "v4_grid_fixed"

Z = 0.01895
D_L_MPC = 82.34
M_BH_MSUN = 6.309573e6
SOURCE_T_K = 30000.0
T_CIRC_DAYS = 0.0

DRIVER_DT_DAYS = 10.0
PRE_PEAK_DAYS = 1200.0
POST_FALLBACK_DAYS = 6500.0

N_RADII = 48
N_GRAINS = 41
N_CLUMPS = 12000
GEOMETRY_SEED = 12345
WISE_TGRID = 900

# -----------------------------------------------------------------------------
# Explicit scientific grid
# -----------------------------------------------------------------------------
# Grain configs are (a_min_um, a_max_um, q).  They are kept as named tuples in
# the Cartesian grid so no impossible a_min >= a_max combinations are created.
# First production grid: broad enough to exercise every major physical block,
# but intentionally not a 10^11-point Cartesian monster.  After the first
# production pass, the same engine can expand any axis without code changes.
GRAIN_CONFIGS = (
    (0.010, 0.25, 3.5),
    (0.010, 1.00, 3.5),
)

GRID = {
    "eta_rad": (0.057, 0.10),
    "tau_visc_days": (0.0, 100.0, 500.0),
    "composition": ("silicate", "graphite"),
    "density_cm3": (1.0e9,),
    "grain_config": GRAIN_CONFIGS,
    "r_in_pc": (0.05, 0.15, 0.30),
    "r_out_pc": (1.0, 3.0),
    "p_radial": (0.0, 1.0, 2.0),
    "theta_open_deg": (60.0,),
    "inclination_deg": (55.0,),
    # Cheap dimensions; do not trigger a new thermal calculation.
    "f_abs_initial": (0.02, 0.05, 0.10, 0.20, 0.40),
    "t0_mjd": tuple(np.arange(56000.0, 58000.1, 20.0)),
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def experiment_manifest():
    files = sorted((PROJECT_ROOT / "src").glob("*.py")) + [Path(__file__), DATA_CSV]
    files += sorted(p for p in (PROJECT_ROOT / "data/reference").rglob("*") if p.is_file())
    files += sorted((STARS_ROOT / "input").rglob("*.dat"))
    hashes = {str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {"version": "V4_FIXED_20260930", "grid": GRID, "hashes": hashes,
            "time_coordinate": "source_retarded_u", "error_model": "input sigma; no added floor",
            "numerics": {"dt_days": DRIVER_DT_DAYS, "n_radii": N_RADII, "n_grains": N_GRAINS,
                         "n_clumps": N_CLUMPS, "seed": GEOMETRY_SEED, "wise_tgrid": WISE_TGRID},
            "source": {"redshift": Z, "D_L_Mpc": D_L_MPC, "M_BH_Msun": M_BH_MSUN,
                       "T_source_K": SOURCE_T_K}}


@lru_cache(maxsize=1)
def experiment_digest():
    return hashlib.sha256(json.dumps(experiment_manifest(),sort_keys=True,default=str).encode()).hexdigest()


def stable_id(payload: dict) -> str:
    raw = json.dumps({"experiment": experiment_digest(), "block": payload}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def discover_stars_files() -> list[Path]:
    root = STARS_ROOT / "input"
    if not root.exists():
        raise FileNotFoundError(f"STARS input directory not found: {root}")
    files = sorted(p for p in root.rglob("*.dat") if p.is_file())
    if not files:
        raise RuntimeError(f"No STARS .dat files found under {root}")
    return files


def load_observations() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Canonical 22-epoch input not found: {DATA_CSV}")
    d = pd.read_csv(DATA_CSV).sort_values("mjd").reset_index(drop=True)
    req = {
        "mjd", "F_W1_Jy", "F_W2_Jy", "sigma_W1_Jy", "sigma_W2_Jy",
        "baseline_W1_mJy", "baseline_W2_mJy",
    }
    missing = sorted(req - set(d.columns))
    if missing:
        raise RuntimeError(f"Missing canonical columns: {missing}")
    if len(d) != 22:
        raise RuntimeError(f"Expected exactly 22 epochs, found {len(d)}")
    a = d[list(req)].to_numpy(float)
    if not np.all(np.isfinite(a)):
        raise RuntimeError("Non-finite value in canonical observational input")
    if np.any(d["sigma_W1_Jy"] <= 0) or np.any(d["sigma_W2_Jy"] <= 0):
        raise RuntimeError("Non-positive observational uncertainty")
    # Baselines must be supplied constants, not re-fitted here.
    if np.ptp(d["baseline_W1_mJy"].to_numpy(float)) > 1e-9:
        raise RuntimeError("W1 baseline is not constant in canonical input")
    if np.ptp(d["baseline_W2_mJy"].to_numpy(float)) > 1e-9:
        raise RuntimeError("W2 baseline is not constant in canonical input")
    return d


def radial_grid_and_weights(rin: float, rout: float, p: float, n: int):
    """Nodes include exact boundaries; weights integrate r^(2-p) dr by cells."""
    r = np.geomspace(rin, rout, int(n))
    edges = np.empty(r.size + 1)
    edges[0], edges[-1] = rin, rout
    edges[1:-1] = np.sqrt(r[:-1] * r[1:])
    expo = 3.0 - p
    if abs(expo) < 1e-12:
        w = np.log(edges[1:] / edges[:-1])
    else:
        w = (edges[1:] ** expo - edges[:-1] ** expo) / expo
    w = np.asarray(w, float)
    w /= w.sum()
    return r, w


def absolute_population_band_cube(
    opacity, bandpass, grains, temperature, alive, q, redshift,
    luminosity_distance_m, vega_wave, vega_flux, n_temperature,
):
    """
    Absolute mean flux density [Jy] per grain population member.

    fast_population_band_cube returns the same Vega convolution of
    a_um^2 Q B_lambda used by the validated shape path.  Convert that to the
    absolute one-grain observer flux used by grain_wise_flux_density_jy:

      a_um^2 -> a_m^2: 1e-12
      L_lambda = 4 pi^2 a_m^2 Q B_lambda
      F_lambda = L_lambda/[4 pi D_L^2 (1+z)]
      SI -> CALSPEC F_lambda: 1e-7

    Therefore the common multiplicative factor is
      pi * 1e-19 / [D_L^2 (1+z)].
    """
    rel = fast_population_band_cube(
        opacity=opacity,
        bandpass=bandpass,
        grain_radius_micron=grains,
        temperature_K=temperature,
        alive=alive,
        q=q,
        redshift=redshift,
        n_temperature=n_temperature,
        vega_wavelength_micron=vega_wave,
        vega_flux_lambda=vega_flux,
    )
    factor = np.pi * 1.0e-19 / (float(luminosity_distance_m) ** 2 * (1.0 + redshift))
    return rel * factor



@lru_cache(maxsize=8)
def _cached_dust_static(composition, amin, amax):
    """Exact cache: opacity, thermal solver, grain grid, validated thermal lookup."""
    opacity = load_dust_opacity(DRAINE_DIR, composition)
    thermal = DustThermalEquilibrium(opacity, source_temperature_K=SOURCE_T_K)
    grains = make_grain_size_grid(float(amin), float(amax), N_GRAINS)
    lookup = build_thermal_lookup(
        thermal_solver=thermal,
        grain_radius_micron=grains,
        Tmin_K=2.0,
        Tmax_K=10000.0,
        n_temperature=1600,
    )
    return opacity, thermal, grains, lookup


@lru_cache(maxsize=1)
def _cached_wise_static():
    w1, w2 = load_wise_w1_w2(WISE_DIR)
    vw, vf = load_calspec_vega(VEGA_FILE)
    return w1, w2, vw, vf


@lru_cache(maxsize=64)
def _cached_geometry(rin, rout, p, theta_open_deg, inclination_deg):
    geom = generate_clumpy_geometry(
        n_clumps=N_CLUMPS,
        r_in_pc=float(rin),
        r_out_pc=float(rout),
        p=float(p),
        theta_open_deg=float(theta_open_deg),
        inclination_deg=float(inclination_deg),
        seed=GEOMETRY_SEED,
    )
    return geom["r_pc"].to_numpy(float), geom["z_obs_pc"].to_numpy(float)


def _temperature_cube_from_cached_lookup(lookup, L_heat, radius, grains,
                                         composition, density_cm3):
    L_heat = np.asarray(L_heat, float)
    radius = np.asarray(radius, float)
    grains = np.asarray(grains, float)
    T_sub = float(sublimation_temperature_K(
        density_cm3=float(density_cm3), composition=composition))
    if T_sub >= lookup.temperature_grid_K[-1]:
        raise ValueError("Lookup ceiling must exceed sublimation threshold")
    Nt, Nr = L_heat.shape
    Na = grains.size
    temperature = np.zeros((Nt, Nr, Na), float)
    alive = np.ones((Nt, Nr, Na), bool)
    flux = incident_flux_W_m2(
        luminosity_erg_s=L_heat, distance_pc=radius[None, :])

    for k in range(Na):
        T_eq = lookup_temperature_from_flux(
            lookup=lookup, incident_flux=flux, grain_index=k)
        high = flux > 10.**lookup.log10_flux_grid_W_m2[k, -1]
        destroy = high | (T_eq >= T_sub)
        destroyed_ever = np.maximum.accumulate(destroy, axis=0)
        alive[:, :, k] = ~destroyed_ever
        T_store = np.asarray(T_eq, float).copy()
        first = destroy & ~np.vstack([
            np.zeros((1, Nr), dtype=bool),
            destroyed_ever[:-1, :]
        ])
        T_store[first] = T_sub
        T_store[destroyed_ever & ~first] = np.nan
        temperature[:, :, k] = T_store
    return temperature, alive, T_sub


def _build_dust_response_cached(time_days, luminosity_erg_s, radius_pc,
                                grains, lookup, composition, density_cm3):
    time = np.asarray(time_days, float)
    luminosity = np.asarray(luminosity_erg_s, float)
    radius = np.asarray(radius_pc, float)
    grains = np.asarray(grains, float)
    # Internal time is u=t_dust-r/c, shared by all radial nodes.
    L_heat = np.broadcast_to(luminosity[:, None], (time.size, radius.size)).copy()
    temperature, alive, T_sub = _temperature_cube_from_cached_lookup(
        lookup, L_heat, radius, grains, composition, density_cm3)
    return DustResponseResult(
        time_days=time, radius_pc=radius, grain_radius_micron=grains,
        luminosity_source_erg_s=luminosity,
        luminosity_heating_erg_s=L_heat,
        temperature_K=temperature, alive=alive,
        sublimation_temperature_K=T_sub,
        composition=composition, density_cm3=float(density_cm3),
    )


def scan_t0_fabs(
    obs, echo_t, echo_w1_ref, echo_w2_ref, f_abs_grid, t0_grid,
    base_w1_jy, base_w2_jy,
):
    """Vectorized cheap grid.  Reference echo is for f_abs=1."""
    mjd = obs["mjd"].to_numpy(float)
    y1 = obs["F_W1_Jy"].to_numpy(float)
    y2 = obs["F_W2_Jy"].to_numpy(float)
    s1 = obs["sigma_W1_Jy"].to_numpy(float)
    s2 = obs["sigma_W2_Jy"].to_numpy(float)

    t0 = np.asarray(t0_grid, float)
    fa = np.asarray(f_abs_grid, float)
    # shape Nt0 x Nepoch
    trel = mjd[None, :] - t0[:, None]
    e1 = np.interp(trel.ravel(), echo_t, echo_w1_ref, left=0.0, right=0.0).reshape(trel.shape)
    e2 = np.interp(trel.ravel(), echo_t, echo_w2_ref, left=0.0, right=0.0).reshape(trel.shape)

    # shape Nf x Nt0 x Nepoch
    m1 = base_w1_jy + fa[:, None, None] * e1[None, :, :]
    m2 = base_w2_jy + fa[:, None, None] * e2[None, :, :]
    c1 = np.sum(((y1[None, None, :] - m1) / s1[None, None, :]) ** 2, axis=2)
    c2 = np.sum(((y2[None, None, :] - m2) / s2[None, None, :]) ** 2, axis=2)
    ct = c1 + c2
    return c1, c2, ct


def physical_block_iter(stars_files):
    keys = (
        "eta_rad", "tau_visc_days", "composition", "density_cm3",
        "grain_config", "r_in_pc", "r_out_pc", "p_radial",
        "theta_open_deg", "inclination_deg",
    )
    vals = [GRID[k] for k in keys]
    for sf in stars_files:
        for combo in itertools.product(*vals):
            d = dict(zip(keys, combo))
            if d["r_out_pc"] <= d["r_in_pc"]:
                continue
            d["stars_file"] = str(sf)
            d["stars_rel"] = str(sf.relative_to(STARS_ROOT))
            yield d


def evaluate_block(block: dict, obs_records: dict) -> dict:
    """Evaluate one expensive physical block and return its best cheap-grid row."""
    t_start = time.perf_counter()
    block_id = stable_id(block)
    try:
        obs = pd.DataFrame(obs_records)
        amin, amax, q = map(float, block["grain_config"])
        rin = float(block["r_in_pc"])
        rout = float(block["r_out_pc"])
        p = float(block["p_radial"])

        _tp = time.perf_counter()
        driver, driver_diagnostics = build_physical_driver(
            stars_file=Path(block["stars_file"]),
            M_BH_Msun=M_BH_MSUN,
            eta_rad=float(block["eta_rad"]),
            tau_visc_days=float(block["tau_visc_days"]),
            t_circ_days=T_CIRC_DAYS,
            dt_days=DRIVER_DT_DAYS,
            pre_peak_days=PRE_PEAK_DAYS,
            post_fallback_days=POST_FALLBACK_DAYS,
        )
        td = driver["t_days"].to_numpy(float)
        L = driver["L_bol_erg_s"].to_numpy(float)
        _t_driver = time.perf_counter() - _tp
        _tp = time.perf_counter()

        opacity, thermal, grains, thermal_lookup = _cached_dust_static(
            block["composition"], amin, amax
        )
        radii, rw = radial_grid_and_weights(rin, rout, p, N_RADII)

        _t_static = time.perf_counter() - _tp
        _tp = time.perf_counter()
        dust = _build_dust_response_cached(
            time_days=td,
            luminosity_erg_s=L,
            radius_pc=radii,
            grains=grains,
            lookup=thermal_lookup,
            composition=block["composition"],
            density_cm3=float(block["density_cm3"]),
        )

        _t_dust = time.perf_counter() - _tp
        _tp = time.perf_counter()
        # f_abs=1 reference normalization.  Every requested f_abs is a strict
        # common linear scaling of grain number after this point.
        pop = build_population_normalization(
            thermal_solver=thermal,
            grain_radius_micron=grains,
            clump_radius_pc=radii,
            f_abs_initial=1.0,
            q=q,
            clump_weight=rw,
        )

        w1, w2, vw, vf = _cached_wise_static()
        _t_population = time.perf_counter() - _tp
        _tp = time.perf_counter()
        dl_m = D_L_MPC * 1.0e6 * PC_M

        pergrain_w1 = absolute_population_band_cube(
            opacity, w1, grains, dust.temperature_K, dust.alive, q, Z,
            dl_m, vw, vf, WISE_TGRID,
        )
        pergrain_w2 = absolute_population_band_cube(
            opacity, w2, grains, dust.temperature_K, dust.alive, q, Z,
            dl_m, vw, vf, WISE_TGRID,
        )
        radial_w1 = pop.total_grain_number * pergrain_w1
        radial_w2 = pop.total_grain_number * pergrain_w2
        _t_wise = time.perf_counter() - _tp
        _tp = time.perf_counter()

        rc, zc = _cached_geometry(
            rin, rout, p,
            float(block["theta_open_deg"]),
            float(block["inclination_deg"]),
        )

        e1, e2 = integrate_radial_responses_over_clumps(
            dust_time_rest_days=td,
            radius_grid_pc=radii,
            radial_responses=np.stack((radial_w1, radial_w2), axis=0),
            clump_radius_pc=rc,
            clump_z_obs_pc=zc,
            redshift=Z,
            chunk_size=256,
            source_time_input=True,
        )

        _t_echo = time.perf_counter() - _tp
        _tp = time.perf_counter()
        b1 = float(obs["baseline_W1_mJy"].iloc[0]) / 1000.0
        b2 = float(obs["baseline_W2_mJy"].iloc[0]) / 1000.0
        c1, c2, ct = scan_t0_fabs(
            obs, e1.time_obs_days, e1.response, e2.response,
            GRID["f_abs_initial"], GRID["t0_mjd"], b1, b2,
        )
        jf, jt = np.unravel_index(np.nanargmin(ct), ct.shape)
        _t_scan = time.perf_counter() - _tp
        best_f = float(GRID["f_abs_initial"][jf])
        best_t0 = float(GRID["t0_mjd"][jt])

        alive_fraction_final = float(np.mean(dust.alive[-1]))
        peak_temp = float(np.nanmax(dust.temperature_K)) if np.any(np.isfinite(dust.temperature_K)) else np.nan

        out = {
            "block_id": block_id,
            "status": "ok",
            "stars_rel": block["stars_rel"],
            "eta_rad": float(block["eta_rad"]),
            "tau_visc_days": float(block["tau_visc_days"]),
            "composition": block["composition"],
            "density_cm3": float(block["density_cm3"]),
            "a_min_micron": amin,
            "a_max_micron": amax,
            "q_grain": q,
            "r_in_pc": rin,
            "r_out_pc": rout,
            "p_radial": p,
            "theta_open_deg": float(block["theta_open_deg"]),
            "inclination_deg": float(block["inclination_deg"]),
            "best_f_abs_initial": best_f,
            "best_t0_mjd": best_t0,
            "chi2_w1": float(c1[jf, jt]),
            "chi2_w2": float(c2[jf, jt]),
            "chi2_total": float(ct[jf, jt]),
            "n_data": 44,
            "time_coordinate": "source_retarded_u",
            "source_energy_erg": driver_diagnostics["radiated_energy_erg"],
            "accretion_mass_ratio": driver_diagnostics["accretion_mass_ratio"],
            "peak_Lbol_erg_s": float(np.nanmax(L)),
            "T_peak_K": peak_temp,
            "alive_grid_fraction_final": alive_fraction_final,
            "peak_w1_days": response_peak_time_days(e1.time_obs_days, e1.response),
            "peak_w2_days": response_peak_time_days(e2.time_obs_days, e2.response),
            "centroid_w1_days": response_centroid_days(e1.time_obs_days, e1.response),
            "centroid_w2_days": response_centroid_days(e2.time_obs_days, e2.response),
            "t_driver_s": float(_t_driver),
            "t_static_s": float(_t_static),
            "t_dust_s": float(_t_dust),
            "t_population_s": float(_t_population),
            "t_wise_s": float(_t_wise),
            "t_echo_s": float(_t_echo),
            "t_scan_s": float(_t_scan),
            "runtime_s": float(time.perf_counter() - t_start),
        }
        out["peak_offset_days"] = out["peak_w2_days"] - out["peak_w1_days"]
        out["centroid_offset_days"] = out["centroid_w2_days"] - out["centroid_w1_days"]
        return out
    except Exception as exc:
        return {
            "block_id": block_id,
            "status": "error",
            "stars_rel": block.get("stars_rel", ""),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8).replace("\n", " | "),
            "runtime_s": float(time.perf_counter() - t_start),
        }


def append_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if row.get("status") == "error":
        with path.with_suffix(".errors.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        return
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path, usecols=["block_id"])["block_id"].astype(str))
    except Exception:
        return set()


def grid_counts(stars_files):
    # Count valid expensive blocks without materializing them.
    n_rad = sum(1 for ri in GRID["r_in_pc"] for ro in GRID["r_out_pc"] if ro > ri)
    n_exp = (
        len(stars_files) * len(GRID["eta_rad"]) * len(GRID["tau_visc_days"])
        * len(GRID["composition"]) * len(GRID["density_cm3"])
        * len(GRID["grain_config"]) * n_rad * len(GRID["p_radial"])
        * len(GRID["theta_open_deg"]) * len(GRID["inclination_deg"])
    )
    n_cheap = len(GRID["f_abs_initial"]) * len(GRID["t0_mjd"])
    return n_exp, n_cheap, n_exp * n_cheap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("count", "benchmark", "production"), default="benchmark")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--max-blocks", type=int, default=8,
                    help="benchmark only; number of expensive physical blocks")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
    args = ap.parse_args()

    stars = discover_stars_files()
    obs = load_observations()
    nexp, ncheap, ntotal = grid_counts(stars)

    print("=" * 78)
    print("UGC 11487 V4 PHYSICAL GRID")
    print(f"STARS curves found       : {len(stars)}")
    print(f"Expensive physical blocks: {nexp:,}")
    print(f"Cheap (t0 x f_abs) / block: {ncheap:,}")
    print(f"Total evaluated models   : {ntotal:,}")
    print(f"Workers requested        : {args.workers}")
    print(f"Observational points     : {2*len(obs)} (22 x 2)")
    print("=" * 78)

    if args.mode == "count":
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTDIR / "manifest.json"
    manifest_text = json.dumps(experiment_manifest(), sort_keys=True, indent=2, default=str)
    if manifest_path.exists() and manifest_path.read_text() != manifest_text:
        raise RuntimeError("Different code/data/grid in this output directory; use a new OUTDIR")
    manifest_path.write_text(manifest_text)

    checkpoint = OUTDIR / ("benchmark_blocks.csv" if args.mode == "benchmark" else "production_blocks.csv")
    if args.fresh and checkpoint.exists():
        raise RuntimeError("--fresh cannot overwrite an existing checkpoint; use a new OUTDIR")
    done = completed_ids(checkpoint)

    blocks = physical_block_iter(stars)
    if args.mode == "benchmark":
        blocks = itertools.islice(blocks, int(args.max_blocks))

    pending = []
    for b in blocks:
        if stable_id(b) not in done:
            pending.append(b)

    if not pending:
        print("Nothing to do: all selected blocks are already in checkpoint.")
        return

    print(f"Pending expensive blocks : {len(pending):,}")
    obs_records = obs.to_dict(orient="list")
    wall0 = time.perf_counter()

    if args.workers <= 1:
        iterator = (evaluate_block(b, obs_records) for b in pending)
        for i, row in enumerate(iterator, 1):
            append_row(checkpoint, row)
            print(f"[{i}/{len(pending)}] {row['status']} {row['block_id']} "
                  f"chi2={row.get('chi2_total', np.nan):.3f} "
                  f"{row['runtime_s']:.1f}s", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(evaluate_block, b, obs_records): b for b in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                append_row(checkpoint, row)
                print(f"[{i}/{len(pending)}] {row['status']} {row['block_id']} "
                      f"chi2={row.get('chi2_total', np.nan):.3f} "
                      f"{row['runtime_s']:.1f}s", flush=True)

    wall = time.perf_counter() - wall0
    print("-" * 78)
    print(f"Wall time                : {wall:.1f} s")
    print(f"Completed blocks         : {len(pending):,}")
    print(f"Aggregate blocks/s       : {len(pending)/wall:.5f}")
    print(f"Equivalent models/s      : {len(pending)*ncheap/wall:.1f}")
    if args.mode == "benchmark" and len(pending) > 0:
        rate = len(pending) / wall
        eta_h = nexp / rate / 3600.0 if rate > 0 else np.inf
        print(f"Naive full-grid ETA      : {eta_h:.2f} h at this benchmark rate")
        print("Run production only after checking benchmark_blocks.csv for errors.")
    print(f"Checkpoint               : {checkpoint}")


if __name__ == "__main__":
    main()
