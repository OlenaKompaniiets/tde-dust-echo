"""V4.6c two-parameter boundary audit.

Tests only the two coordinates that hit the V4.6b upper grid boundaries:
compact Rin and extended direct optical-depth parameter tau_E.

Everything else is fixed at the V4.6b best point:
    compact Rout=0.50 pc, p=0
    extended Rin=1.00 pc, Rout=2.00 pc, p=2
    tau_C=0.40
    ir_fraction=0.50

Run from project root:
    python scripts/run_v46c_boundary_audit.py --workers 16

This is a 5 x 4 = 20 block audit, with checkpoint/resume.
"""

from __future__ import annotations

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
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


OUT = ROOT / "results" / "v46c_boundary_audit"
CHECKPOINT = OUT / "blocks.csv"

# Only the two dimensions that hit the V4.6b upper boundaries.
RIN_C = (0.20, 0.25, 0.30, 0.35, 0.40)
TAU_E = (2.20, 2.60, 3.00, 3.40)


def make_config(rc, te):
    c = dict(v46.BASE)

    # Fixed at / around the V4.6b best solution.
    c["compact"] = (float(rc), 0.50, 0.0)
    c["extended"] = (1.00, 2.00, 2.0)

    c["tau_C"] = 0.40
    c["tau_E"] = float(te)

    # Physical isotropic outward-hemisphere approximation.
    c["ir_fraction"] = 0.50

    c["label"] = "v46c_boundary_audit"

    return c


def task_key(rc, te):
    return (
        round(float(rc), 8),
        round(float(te), 8),
    )


def worker(c):
    r = v46.evaluate(c)
    e = r["energy"]

    return {
        "compact_Rin_pc": float(c["compact"][0]),
        "compact_Rout_pc": 0.50,
        "compact_p": 0.0,

        "extended_Rin_pc": 1.00,
        "extended_Rout_pc": 2.00,
        "extended_p": 2.0,

        "tau_C": 0.40,
        "tau_E": float(c["tau_E"]),
        "ir_fraction": 0.50,

        "chi2_total": float(r["chi2_total"]),
        "chi2_w1": float(r["chi2_w1"]),
        "chi2_w2": float(r["chi2_w2"]),
        "t0_mjd": float(r["t0_mjd"]),

        "secondary_to_direct_absorbed_ratio":
            float(e["secondary_to_direct_absorbed_ratio"]),

        "energy_budget_relative_max":
            float(e["energy_budget_relative_max"]),

        "tau_3p4":
            float(e["initial_radial_IR_absorption_depth"]["3.4"]),

        "tau_4p6":
            float(e["initial_radial_IR_absorption_depth"]["4.6"]),

        "tau_10":
            float(e["initial_radial_IR_absorption_depth"]["10.0"]),
    }


def save(rows):
    df = pd.DataFrame(rows)

    if len(df):
        df = df.sort_values(
            ["chi2_total", "compact_Rin_pc", "tau_E"],
            kind="stable",
        ).reset_index(drop=True)

    df.to_csv(CHECKPOINT, index=False)

    return df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Resume from checkpoint if it already exists.
    # ------------------------------------------------------------

    if CHECKPOINT.exists():
        old = pd.read_csv(CHECKPOINT)
    else:
        old = pd.DataFrame()

    if len(old):
        rows = old.to_dict("records")

        done = {
            task_key(
                row["compact_Rin_pc"],
                row["tau_E"],
            )
            for _, row in old.iterrows()
        }

    else:
        rows = []
        done = set()

    all_tasks = [
        make_config(rc, te)
        for rc in RIN_C
        for te in TAU_E
    ]

    tasks = [
        c
        for c in all_tasks
        if task_key(
            c["compact"][0],
            c["tau_E"],
        ) not in done
    ]

    n_total = len(RIN_C) * len(TAU_E)

    print("=" * 78)
    print("V4.6c BOUNDARY AUDIT")
    print("=" * 78)

    print(
        "fixed: extended=(1.00,2.00,p=2), "
        "tau_C=0.40, ir_fraction=0.50"
    )

    print("compact Rin =", RIN_C)
    print("tau_E       =", TAU_E)

    print("total blocks =", n_total)
    print("done         =", len(done))
    print("remaining    =", len(tasks))
    print("workers      =", args.workers)
    print("checkpoint   =", CHECKPOINT)

    print(flush=True)

    # ------------------------------------------------------------
    # Parallel calculation
    # ------------------------------------------------------------

    errors = []

    if tasks:

        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:

            futures = {
                executor.submit(worker, c): c
                for c in tasks
            }

            nnew = 0

            for future in as_completed(futures):

                c = futures[future]

                try:

                    r = future.result()

                    rows.append(r)

                    nnew += 1

                    # Checkpoint after every successful expensive block.
                    save(rows)

                    print(
                        f"[{len(done) + nnew:2d}/{n_total}] "
                        f"RinC={r['compact_Rin_pc']:.2f} "
                        f"tauE={r['tau_E']:.2f} "
                        f"chi2={r['chi2_total']:.6f} "
                        f"W1={r['chi2_w1']:.3f} "
                        f"W2={r['chi2_w2']:.3f} "
                        f"tauW1={r['tau_3p4']:.3f} "
                        f"tauW2={r['tau_4p6']:.3f}",
                        flush=True,
                    )

                except Exception:

                    err = {
                        "config": c,
                        "traceback": traceback.format_exc(),
                    }

                    errors.append(err)

                    print(
                        err["traceback"],
                        flush=True,
                    )

    # ------------------------------------------------------------
    # Final tables
    # ------------------------------------------------------------

    df = save(rows)

    if errors:

        (
            OUT / "errors.json"
        ).write_text(
            json.dumps(
                errors,
                indent=2,
            ),
            encoding="utf-8",
        )

    if len(df) == 0:
        raise RuntimeError(
            "No successful V4.6c blocks"
        )

    best = df.iloc[0].to_dict()

    # ------------------------------------------------------------
    # Profile minima
    # ------------------------------------------------------------

    profile_rc = (
        df
        .groupby(
            "compact_Rin_pc",
            as_index=False,
        )["chi2_total"]
        .min()
        .sort_values(
            "compact_Rin_pc"
        )
    )

    profile_tau = (
        df
        .groupby(
            "tau_E",
            as_index=False,
        )["chi2_total"]
        .min()
        .sort_values(
            "tau_E"
        )
    )

    profile_rc.to_csv(
        OUT / "profile_compact_Rin.csv",
        index=False,
    )

    profile_tau.to_csv(
        OUT / "profile_tau_E.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Explicit boundary checks
    # ------------------------------------------------------------

    rc_boundary = (
        np.isclose(
            best["compact_Rin_pc"],
            min(RIN_C),
        )
        or
        np.isclose(
            best["compact_Rin_pc"],
            max(RIN_C),
        )
    )

    tau_boundary = (
        np.isclose(
            best["tau_E"],
            min(TAU_E),
        )
        or
        np.isclose(
            best["tau_E"],
            max(TAU_E),
        )
    )

    # ------------------------------------------------------------
    # Save manifest
    # ------------------------------------------------------------

    manifest = {

        "experiment":
            "V4.6c two-parameter boundary audit",

        "fixed": {

            "compact_Rout_pc": 0.50,
            "compact_p": 0.0,

            "extended_Rin_pc": 1.00,
            "extended_Rout_pc": 2.00,
            "extended_p": 2.0,

            "tau_C": 0.40,

            "ir_fraction": 0.50,
        },

        "compact_Rin_pc_grid":
            list(RIN_C),

        "tau_E_grid":
            list(TAU_E),

        "n_expected":
            n_total,

        "n_success":
            int(len(df)),

        "best":
            best,

        "best_compact_Rin_on_boundary":
            bool(rc_boundary),

        "best_tau_E_on_boundary":
            bool(tau_boundary),

        "instruction":
            (
                "Do not extend automatically if either "
                "best coordinate remains on a boundary; "
                "inspect profiles and physical diagnostics first."
            ),
    }

    (
        OUT / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        OUT / "best.json"
    ).write_text(
        json.dumps(
            best,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("BEST V4.6c")
    print("=" * 78)

    for k in (

        "chi2_total",
        "chi2_w1",
        "chi2_w2",
        "t0_mjd",

        "compact_Rin_pc",
        "tau_E",

        "secondary_to_direct_absorbed_ratio",

        "tau_3p4",
        "tau_4p6",
        "tau_10",

        "energy_budget_relative_max",
    ):

        print(
            f"{k:36s} = {best[k]}"
        )

    print(
        "compact Rin boundary hit            =",
        bool(rc_boundary),
    )

    print(
        "tau_E boundary hit                  =",
        bool(tau_boundary),
    )

    print()

    print(
        "Profile min chi2 vs compact Rin:"
    )

    print(
        profile_rc.to_string(
            index=False
        )
    )

    print()

    print(
        "Profile min chi2 vs tau_E:"
    )

    print(
        profile_tau.to_string(
            index=False
        )
    )

    print()

    print(
        "saved to",
        OUT,
    )


if __name__ == "__main__":
    main()