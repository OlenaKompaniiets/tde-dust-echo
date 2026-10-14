# V4.8 refined Case A/B: conditional intervals and direct checks

Published 14 October 2026 as a versioned evidence-and-source supplement. This extends the archived V4.8 example without replacing the physical solver, old Case A visualisations or interactive website. It is not a new one-command runnable release.

## Browse the material

- [Case A reference](results/case_a/reference/) and [12 checked endpoints](results/case_a/endpoints/).
- [Case B reference](results/case_b/reference/), [12 checked endpoints](results/case_b/endpoints/) and [four resolution tests](results/case_b/resolution/).
- [Exact endpoint values and source identities](tables/verified_endpoint_values.csv), [refined reference identities](tables/refined_reference_identities.json).
- [Original source scripts](source/), including [A profile update](source/V48_PROFILE_UPDATE/update_profiles.py), [A direct validation](source/V48_PROFILE_FINAL_CHECK/run_validation.py), and B [refinement](source/run_case_b_refine453.py), [targeted checks](source/run_case_b_targeted468.py), [resolution checks](source/run_case_b_resolution485.py).
- [Method and limitations](../../docs/fitting.html#uncertainty), [file hashes and sanitisation record](MANIFEST.json).

## Result identities and interval interpretation

The adopted direct references have chi-square 58.83195387879388 (Case A) and 58.923309117058764 (Case B). The original archived website example remains at 58.95064072368716. These are different saved parameter configurations within V4.8, not changes to its physical equations.

The twelve selected physical-parameter endpoints per case have direct delta-chi-square ranges 0.957856–1.041510 (A) and 0.907753–1.009744 (B). In B, ten endpoints come from REFINE453; the upper eta and lower inclination are replaced by the later TARGETED468 checks. The table retains exact source identities.

These approximate conditional intervals describe parameter variation within the adopted physical and numerical assumptions. They are not calibrated comprehensive 1-sigma uncertainties. Other parameters and the epoch were optimised on an approximation to stored RT responses; direct endpoint checks reoptimised the epoch with the other physical parameters fixed. The original reports explicitly set confidence_intervals_certified to false. Case B resolution checks change one numerical setting at the fixed physical reference; they do not establish convergence of interval endpoints.

## Source and reproducibility boundary

Sources are preserved with their original bytes. CASE_B_438_PROFILES is explicitly the 438-model stage, not a final 485-model profile-regeneration pipeline. Later fixed-plan runners embed the exact executed plans. Their embedded plans were checked against saved validation plans (15 refinement, 19 targeted and 5 resolution configurations).

This supplement is not an installed project root. Historical scripts expect project-root placement, V48_PAPER_FINAL and V48_LOCAL_PAIR, baseline cache/seed files and the original results layout. Later B runners import cache_utils, budget and rt_queue from V48_CASE_B_CHECK. Full profile regeneration additionally needs large response caches and observation arrays not included in this small supplement. The recovered original generator of the later B profile plans has not been identified; replay of saved direct checks is not regeneration of the entire inference pipeline.

Runtime safeguards include Python 3.11.16 / NumPy 2.4.6 and source/data fingerprints. Do not remove safeguards or rewrite provenance merely to make another environment pass. A complete tested environment/cache-restoration recipe remains required before claiming end-to-end reproduction. No new fit, RT computation or performance benchmark was run for this publication; static source/JSON/CSV checks were performed.

## Capabilities and possible further development

Existing workflows support saved-result/cache reuse and controlled parallel evaluation. The refined analysis was organised within a finite single-PC computation budget. Further optimisation, scaling and expanded direct/profile/coverage validation are potential work, not completed or benchmarked improvements. More hardware alone does not establish statistical coverage, resolution convergence or a more complete physical model; no cluster requirement is asserted.

## Provenance handling

MANIFEST.json gives source-artifact identity and original/published SHA256. In provenance JSON only, absolute machine-path prefixes were replaced with <PROJECT_ROOT> (or <ORIGINAL_WORKSPACE> where applicable). Scientific values and recorded source/input hashes are unchanged. The stored run fingerprint remains the original run fingerprint and must not be recomputed from this sanitised representation. Original absolute paths are not published in this supplement.
