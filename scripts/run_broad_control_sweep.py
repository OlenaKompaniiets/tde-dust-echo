"""
Broad validation sweep for UGC 11487 V4_FIXED.

Purpose:
- exercise all major repaired physical branches before full production;
- NOT a parameter-inference run;
- uses the exact production evaluator in scripts/run_v4_model_grid.py.

Grid:
4 representative STARS drivers
x 3 tau_visc
x 2 compositions
x 3 Rin
x 2 p
x 2 grain ranges
= 288 expensive physical blocks.

Fixed for this validation:
eta=0.10, Rout=1 pc, nH=1e9 cm^-3, theta_open=60 deg, inclination=55 deg.

Output:
results/validation/broad_control_sweep.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v4_model_grid as g

OUTDIR = ROOT / "results" / "validation"
OUTCSV = OUTDIR / "broad_control_sweep.csv"
MANIFEST = OUTDIR / "broad_control_sweep_manifest.json"

STARS = (
    "input/m0.3_t0.0/0.900.dat",
    "input/m1.0_t0.57/2.000.dat",
    "input/m1.0_t1.0/2.000.dat",
    "input/m3.0_t0.0/1.000.dat",
)
TAU = (0.0, 100.0, 500.0)
COMPOSITIONS = ("silicate", "graphite")
RIN = (0.05, 0.15, 0.30)
P = (0.0, 2.0)
GRAINS = ((0.01, 0.25, 3.5), (0.01, 1.00, 3.5))


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        d = pd.read_csv(path, usecols=["block_id"])
        return set(d["block_id"].astype(str))
    except Exception:
        return set()


def blocks():
    for rel in STARS:
        sf = g.STARS_ROOT / rel
        if not sf.exists():
            raise FileNotFoundError(f"Missing STARS curve: {sf}")
        for tau in TAU:
            for comp in COMPOSITIONS:
                for rin in RIN:
                    for p in P:
                        for grain in GRAINS:
                            yield dict(
                                eta_rad=0.10,
                                tau_visc_days=tau,
                                composition=comp,
                                density_cm3=1.0e9,
                                grain_config=grain,
                                r_in_pc=rin,
                                r_out_pc=1.0,
                                p_radial=p,
                                theta_open_deg=60.0,
                                inclination_deg=55.0,
                                stars_file=str(sf),
                                stars_rel=rel,
                            )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_blocks = list(blocks())
    done = existing_ids(OUTCSV)

    manifest = {
        "purpose": "broad pre-production validation; not inference",
        "n_blocks": len(all_blocks),
        "stars": STARS,
        "tau_visc_days": TAU,
        "compositions": COMPOSITIONS,
        "r_in_pc": RIN,
        "r_out_pc": 1.0,
        "p_radial": P,
        "grain_configs": GRAINS,
        "eta_rad": 0.10,
        "density_cm3": 1.0e9,
        "theta_open_deg": 60.0,
        "inclination_deg": 55.0,
        "production_experiment_digest": g.experiment_digest(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    obs = g.load_observations().to_dict(orient="list")
    print("=" * 78)
    print("V4_FIXED BROAD CONTROL SWEEP")
    print(f"Total blocks : {len(all_blocks)}")
    print(f"Already done : {len(done)}")
    print(f"Output       : {OUTCSV}")
    print("=" * 78, flush=True)

    t0 = time.perf_counter()
    nok = 0
    nerr = 0

    for i, b in enumerate(all_blocks, start=1):
        bid = g.stable_id(b)
        if bid in done:
            continue

        row = g.evaluate_block(b, obs)
        g.append_row(OUTCSV, row)

        if row.get("status") == "ok":
            nok += 1
            print(
                f"[{i:03d}/{len(all_blocks)}] ok "
                f"{b['stars_rel']} tau={b['tau_visc_days']:.0f} "
                f"{b['composition']} Rin={b['r_in_pc']:.2f} "
                f"p={b['p_radial']:.0f} amax={b['grain_config'][1]:.2f} "
                f"chi2={row['chi2_total']:.3f} "
                f"alive={row['alive_grid_fraction_final']:.3f} "
                f"mass={row['accretion_mass_ratio']:.12f} "
                f"{row['runtime_s']:.1f}s",
                flush=True,
            )
        else:
            nerr += 1
            print(
                f"[{i:03d}/{len(all_blocks)}] ERROR "
                f"{b['stars_rel']} tau={b['tau_visc_days']:.0f}: "
                f"{row.get('error')}",
                flush=True,
            )

    if OUTCSV.exists():
        d = pd.read_csv(OUTCSV)
        print("=" * 78)
        print("BROAD CONTROL COMPLETE")
        print(f"Successful rows : {len(d)}")
        print(f"Errors this run : {nerr}")
        print(f"Minimum chi2    : {d['chi2_total'].min():.6f}")
        print(f"Max |Mratio-1|  : {(d['accretion_mass_ratio'] - 1.0).abs().max():.3e}")
        print(f"Min alive frac  : {d['alive_grid_fraction_final'].min():.6f}")
        print(f"Wall time       : {(time.perf_counter()-t0)/60:.1f} min")
        print(f"Saved           : {OUTCSV}")
        print("=" * 78)


if __name__ == "__main__":
    main()
