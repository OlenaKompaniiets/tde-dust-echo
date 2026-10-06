"""V4.6b focused geometry/optical-depth refit with fixed physical IR split.

Purpose
-------
Refit a deliberately small neighbourhood around the validated V4.3c/V4.5
solution using the validated V4.6 direct + outward diffuse-IR solver.

Physics held fixed:
    ir_reprocess_fraction = 0.5
    composition/grains/driver/eta and all other BASE settings inherited
    from scripts.run_v46_ir_reprocessing.

Varied:
    compact Rin
    extended Rin, Rout, p
    tau_C, tau_E

The script uses the unchanged V4.6 evaluate() path, so every block retains
the same direct-transfer, diffuse-IR, WISE, observer-delay, and t0 fitting
implementation as the validated V4.6 control.

Run from project root:
    python scripts/run_v46b_focused_refit.py --workers 16

Checkpoint/resume:
    results/v46b_focused_refit/blocks.csv
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import sys
import json
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_v46_ir_reprocessing as v46


OUT = ROOT / "results" / "v46b_focused_refit"
CHECKPOINT = OUT / "blocks.csv"
ERRORS = OUT / "errors.json"


# Focused, physically motivated neighbourhood.  Do not expand this grid
# automatically if the best solution lands on an edge; inspect first.
COMPACT_RIN = (0.15, 0.20, 0.25)
EXTENDED = (
    (0.75, 1.50, 1.0),
    (0.75, 2.00, 1.0),
    (1.00, 1.50, 1.0),
    (1.00, 1.50, 2.0),
    (1.00, 2.00, 1.0),
    (1.00, 2.00, 2.0),
)
TAU_C = (0.25, 0.40, 0.60)
TAU_E = (1.40, 1.80, 2.20, 2.60)


def make_tasks():
    tasks = []
    for rc in COMPACT_RIN:
        for re, ro, pe in EXTENDED:
            for tc in TAU_C:
                for te in TAU_E:
                    c = dict(v46.BASE)
                    # Preserve the V4.3c compact outer edge and p=0.
                    c["compact"] = (float(rc), 0.50, 0.0)
                    c["extended"] = (float(re), float(ro), float(pe))
                    c["tau_C"] = float(tc)
                    c["tau_E"] = float(te)
                    c["ir_fraction"] = 0.5
                    c["label"] = "v46b_focused_refit"
                    tasks.append(c)
    return tasks


def key(c):
    return (
        round(float(c["compact"][0]), 8),
        round(float(c["extended"][0]), 8),
        round(float(c["extended"][1]), 8),
        round(float(c["extended"][2]), 8),
        round(float(c["tau_C"]), 8),
        round(float(c["tau_E"]), 8),
    )


def worker(c):
    r = v46.evaluate(c)
    en = r["energy"]
    return {
        "compact_Rin_pc": c["compact"][0],
        "compact_Rout_pc": c["compact"][1],
        "compact_p": c["compact"][2],
        "extended_Rin_pc": c["extended"][0],
        "extended_Rout_pc": c["extended"][1],
        "extended_p": c["extended"][2],
        "tau_C": c["tau_C"],
        "tau_E": c["tau_E"],
        "ir_fraction": c["ir_fraction"],
        "chi2_total": r["chi2_total"],
        "chi2_w1": r["chi2_w1"],
        "chi2_w2": r["chi2_w2"],
        "t0_mjd": r["t0_mjd"],
        "E_source_window_erg": en["E_source_window_erg"],
        "E_direct_absorbed_erg": en["E_direct_absorbed_erg"],
        "E_secondary_IR_absorbed_erg": en["E_secondary_IR_absorbed_erg"],
        "secondary_to_direct_absorbed_ratio": en["secondary_to_direct_absorbed_ratio"],
        "energy_budget_relative_max": en["energy_budget_relative_max"],
        "tau_3p4": en["initial_radial_IR_absorption_depth"]["3.4"],
        "tau_4p6": en["initial_radial_IR_absorption_depth"]["4.6"],
        "tau_10": en["initial_radial_IR_absorption_depth"]["10.0"],
    }


def load_done():
    if not CHECKPOINT.exists():
        return pd.DataFrame()
    df = pd.read_csv(CHECKPOINT)
    return df


def row_key(row):
    return (
        round(float(row["compact_Rin_pc"]), 8),
        round(float(row["extended_Rin_pc"]), 8),
        round(float(row["extended_Rout_pc"]), 8),
        round(float(row["extended_p"]), 8),
        round(float(row["tau_C"]), 8),
        round(float(row["tau_E"]), 8),
    )


def save(rows):
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("chi2_total", kind="stable").reset_index(drop=True)
    df.to_csv(CHECKPOINT, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    OUT.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks()
    old = load_done()
    rows = old.to_dict("records") if len(old) else []
    done = {row_key(x) for _, x in old.iterrows()} if len(old) else set()
    todo = [c for c in tasks if key(c) not in done]

    print("=" * 78)
    print("V4.6b FOCUSED REFIT")
    print("=" * 78)
    print("fixed ir_fraction = 0.5")
    print("total blocks      =", len(tasks))
    print("already complete  =", len(done))
    print("remaining         =", len(todo))
    print("workers           =", args.workers)
    print("checkpoint        =", CHECKPOINT)
    print(flush=True)

    errors = []
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(worker, c): c for c in todo}
            completed_now = 0
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    r = fut.result()
                    rows.append(r)
                    completed_now += 1
                    print(
                        f"[{len(done)+completed_now:4d}/{len(tasks)}] "
                        f"Rc={r['compact_Rin_pc']:.2f} "
                        f"Re={r['extended_Rin_pc']:.2f}-{r['extended_Rout_pc']:.2f} "
                        f"pE={r['extended_p']:.0f} "
                        f"tauC={r['tau_C']:.2f} tauE={r['tau_E']:.2f} "
                        f"chi2={r['chi2_total']:.4f} "
                        f"W1={r['chi2_w1']:.2f} W2={r['chi2_w2']:.2f}",
                        flush=True,
                    )
                    # Checkpoint after every successful expensive block.
                    save(rows)
                except Exception:
                    msg = {
                        "config": c,
                        "traceback": traceback.format_exc(),
                    }
                    errors.append(msg)
                    print(msg["traceback"], flush=True)

    df = save(rows)
    if errors:
        ERRORS.write_text(json.dumps(errors, indent=2), encoding="utf-8")

    if len(df) == 0:
        raise RuntimeError("No successful V4.6b blocks")

    best = df.iloc[0].to_dict()
    (OUT / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    # Compact manifest: enough to reproduce and audit the search.
    manifest = {
        "experiment": "V4.6b focused geometry/optical-depth refit",
        "fixed_ir_reprocess_fraction": 0.5,
        "compact_Rin_pc": list(COMPACT_RIN),
        "compact_Rout_pc": 0.50,
        "compact_p": 0.0,
        "extended_configs": [list(x) for x in EXTENDED],
        "tau_C": list(TAU_C),
        "tau_E": list(TAU_E),
        "n_blocks": len(tasks),
        "n_success": int(len(df)),
        "n_errors_this_run": len(errors),
        "best": best,
        "note": (
            "Focused validation grid only. If a minimum lies on a tested boundary, "
            "do not automatically extend the grid; inspect residuals, optical depths, "
            "energy closure, and physical validity first."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("BEST V4.6b")
    print("=" * 78)
    for k in (
        "chi2_total", "chi2_w1", "chi2_w2", "t0_mjd",
        "compact_Rin_pc", "extended_Rin_pc", "extended_Rout_pc", "extended_p",
        "tau_C", "tau_E", "secondary_to_direct_absorbed_ratio",
        "tau_3p4", "tau_4p6", "tau_10", "energy_budget_relative_max",
    ):
        print(f"{k:36s} = {best[k]}")
    print("saved to", OUT)


if __name__ == "__main__":
    main()
