# Release preparation audit

## Input archive

UGC11487_V4_github.zip contained 468 files, including multiple working copies and backups. No canonical input photometry and no STARS input library were included in that archive.

The canonical photometry and STARS Case A curve were restored from the earlier project files. Both match the final archived provenance **byte for byte**:

- Photometry SHA-256: 11a564d90038aceba604f4cfd78388026836a6ee0836ee9526d683ec2061b752
- input/m3.0_t0.0/0.850.dat SHA-256: 84b0f6e031d110ae04bd2776890bfeb8e2460e4cc967f2afcd8e22963d55650b

The 141 STARS histories were copied from the earlier project input library. All shipped input hashes are listed in release_manifest.json. Only the Case A curve and photometry are claimed here to have been matched to the final Case A provenance.

## Scientific implementation

All 62 inherited Python files in the selected module/workflow directories are unchanged from the supplied archive; see source_comparison.json. No thermal or transport algorithm was modified. New tools provide dependency checks, a saved-result check, a timestamped Case A launch, and corrected thermal visualization. Backup copies, compiled Python caches, and duplicate images were omitted.

The final solver is an effective axisymmetric static-opacity model, not the early clumpy-grain-destruction V4 model. A sublimation gate is not a time-dependent destruction solver. These distinctions are explicit in the website.

## Reproduction

A fresh isolated Python 3.12 environment was installed from requirements.txt. The Case A solver completed and the corrected plotting script completed. See VALIDATION_RELEASE.json and validation/reproduction.log for numerical results. requirements-tested-py312.txt records exact installed packages. This does not validate Windows or all configurations. The standard environment.yml targets the user's Python 3.11 workflow and resolves compatible package versions independently.

Input hashes, source parsing, archived chi-square reconstruction, CLI help, and website local links were checked. Browser screenshot validation could not be completed because the browser download returned an invalid archive. Review the visual layout locally before publishing.

## Remaining release decisions

- Confirm author list, copyright and software license.
- Confirm redistribution terms for third-party inputs or replace them with verified download instructions.
- Name the GitHub repository and complete CITATION.cff.template.
- Review the website in a browser and publish the prepared /docs directory.
- Add final validated uncertainty intervals when available.

No GitHub repository was created or modified, no site was deployed, and no DOI was assigned.
