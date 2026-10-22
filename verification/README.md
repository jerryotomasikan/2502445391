# Prediction-level verification data for Paper 694

Version 1.1.0. This release provides saved per-observation predictions and model-selection records for 33 recorded runs: 19 principal configurations, four duplicate-retaining arms, and ten red-wine seeded runs (including separately saved primary-seed runs). No fitted models or upstream raw source datasets are redistributed.

## Contents

- evidence/<run>/: original saved binary and ordinal predictions, development selection records, metrics, bootstrap comparison summaries, development/test and CV-fold indices, data audits and original software-version contexts. baseline_predictions.csv is newly derived solely from each run's development target values.
- lineage/<run>_rows.csv: every analyzed source row before duplicate removal, with zero-based original CSV data-row identifier, analysis-row identifier, exact-predictor duplicate group, retained/discarded status, clean-row identifier, transformed ordinal/binary targets, partition and binary/ordinal validation-fold IDs. -1 denotes discarded rows or no development fold. CV NPZ indices are relative to development; lineage fold IDs have already been mapped to clean/source rows.
- lineage/<run>_targets.json: reconstructed exact full-precision bin edges, original bin-to-final-grade mapping, binary threshold/operator, original specifications, seed and development-selected constant. Native ordinal targets have null bin edges.
- results/: original aggregate tables used in the article and original public source manifest.
- documentation/: schemas, hashes, provenance, validation reports, original recorded package versions, and checkpoint availability.
- scripts/verify_predictions.py: portable prediction-level metric and conditional-bootstrap verifier; no training or original datasets required.
- scripts/reconstruct_lineage.py: optional reconstruction from original working CSVs; requires --dataset-dir pointing to files whose hashes match results/source_manifest.json. Fails if clean checksums differ. No model training.
- scripts/original_target_construction.py: original notebook target cell retained for provenance; the reconstruction tool imports only its function definitions using AST, not notebook execution.

## Reproduce the verification

Use Python 3.12 with numpy, pandas, scipy and scikit-learn. See requirements-verification.txt for the versions actually used for the new verification, and documentation/original_package_versions.json for the separately recorded training environments. The supplied validation environment is not claimed to be the original training environment.

```bash
python -m pip install -r requirements-verification.txt
python scripts/verify_predictions.py
python scripts/verify_published_tables.py
# Optional: regenerate source lineage using the original hash-matched input CSVs.
python scripts/reconstruct_lineage.py --dataset-dir /path/to/Dataset
```

The validation reports record which checks actually ran. Unchanged source hashes and reconstructed clean-data hashes matched for all 33 runs. Source-row mappings and exact bin edges were reconstructed after the original experiments and verified against original clean-data hashes and holdout labels; they are not misrepresented as originally saved fields. The source features themselves are not redistributed.

Scripts that work on stored predictions do not fit any model. Full retraining remains documented in the parent repository's notebooks/ and runner/ code and requires upstream source files, training dependencies, compute and original TabPFN weights where available.

## Limits

Target boundaries use all analyzed labels before splitting; protocols use different test populations; task variants are dependent; search budgets differ; the analysis is retrospective; bootstrap intervals condition on fitted predictions; and five-seed coverage is red wine only. The air-quality source's sampling and label-generation provenance remains unresolved. Numerical identity to a public source does not resolve original provenance.

Automatic TabPFN checkpoint resolution was used. Exact original checkpoint identifiers and weight-file checksums were not recorded and no checkpoint files were found in this backup. They cannot be inferred from package versions. Checkpoint absence restricts exact retraining, but the released saved probabilities and predictions allow prediction-based verification without a checkpoint.

## Attribution and licences

Authors: Heriyanto and Suharjito. New computational prediction outputs, reconstructed lineage and documentation are released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Verification software is MIT. Original upstream datasets are not relicensed or redistributed and remain subject to provider terms. The older Zenodo software archive has its own MIT record-level licence and is distinct from this release.

The authors used AI-assisted code and manuscript preparation and remain responsible for the released records and interpretation. No new experiment runs or trained model results were invented.

New verification results: 594 source-lineage/split/fold checks, 1,332 saved-prediction/metric/bootstrap checks, and 675 original published-table checks passed (2,601 total; zero failures). The two baseline-seed runs are retained both as principal records and as separately saved seed-sweep records for provenance, not counted as independent replications.
