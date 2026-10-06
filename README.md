# UGC 11487 time-dependent dust echo

Research software and static documentation for the V4.8 effective-medium dust radiative-transfer model. This is a **release-preparation candidate (0.1.0-rc1)**, not a published release.

## Start here

```console
conda env create -f environment.yml
conda activate ugc-dust
python tools/check_install.py
python tools/check_saved_case.py
python -u tools/run_case_a.py --plots
```

Run commands from the repository root. The last command runs one model and writes to a new timestamped `results/case_a_*` folder. It does not launch a grid.

The input/configuration bundle reproduces the archived Case A. The reference chi-square is 58.95064072368716, with nominal conditional 37 degrees of freedom. See `VALIDATION_RELEASE.json` for the actual check in a new environment; environment and numerical differences must be distinguished from parameter changes.

## Documentation website

Open `docs/index.html`, or run:

```console
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. For GitHub Pages, select the `main` branch and `/docs` folder in repository Settings → Pages. The site uses local assets and no build dependencies. No site has been deployed by this preparation.

## Main capabilities

- STARS fallback scaling and causal accretion smoothing.
- Time-, position-, and grain-size-dependent radiative equilibrium.
- Two disjoint radial populations in an axisymmetric effective medium.
- Three-dimensional path delays, thermal secondary heating, observer transfer.
- Absolute emission normalization, WISE equal-energy synthetic photometry, fixed-host likelihood.
- Saved thermal arrays, energy ledger, input fingerprints, and scientific plots.

## Scope

The production solver uses static opacity and a sublimation threshold gate. It does not evolve grain destruction. Scattering and steady background heating are omitted. The code retains UGC-specific constants, a 22-epoch input requirement, and a bounded event-epoch search; see the input guide before applying it to another object. A plotted 3D sampling point is not an individually resolved physical clump.

## Layout

- `tools/`: public example commands and corrected thermal plots.
- `examples/case_a/`: archived scientific products.
- `configs/case_a.json`: exact reference physical configuration.
- `V48_LOCAL_PAIR/solver.py`: unchanged archived fine solver.
- `src/`, `scripts/`: physical modules and inherited workflows.
- `V48_PAPER_FINAL/`, `V48_UNCERTAINTIES/`: advanced project-specific workflows; require baseline/checkpoint setup.
- `docs/`: complete static documentation website.
- `release_manifest.json`: scientific-input hashes and validation environment.

The legacy scripts are retained because the solver imports several of them. Their old version numbers are not the recommended entry point. Existing old thermal plots are superseded by the output of `tools/plot_thermal.py`.

## Release status

Read `AUDIT.md`, `THIRD_PARTY.md`, and `PUBLISH_UA.md` before publishing. The repository owner selected CC0-1.0 in the initial GitHub commit; that LICENSE is preserved. The software author list still needs maintainer confirmation. A template is provided for the citation record; no authors, software DOI, or repository URL have been invented. Final uncertainty intervals are not included because completed validated profiles were not supplied.
