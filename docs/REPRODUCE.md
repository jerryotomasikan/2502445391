# Reproducing the paper

Three levels, in increasing cost. Each says what it needs and what it verifies.

---

## Level 1 — Read the recorded results

**Needs:** nothing beyond this repository.

`results/` holds the outputs behind every table:

| File | Backs |
|---|---|
| `audited_results.csv` | Tables I, S2; per-configuration model selection and metrics |
| `protocol_comparison.csv` | Table II |
| `ordinal_contrasts.csv` | Table III |
| `category_bootstrap.csv` | Tables IV, S1 |
| `ordinal_error_accounting.csv` | Table S3 |
| `primary_runner_inference.csv` | Table S4 |
| `rejection_boundary.csv` | Table S5 |
| `seed_category_differences.csv` | Paired category differences across the five red-wine seeds |
| `seed_comparison.csv`, `seed_resolution.csv` | Seed sweep, Figure 2(b) |
| `effort_versus_protocol.csv` | Modelling stage against the duplicate decision |
| `ensemble_ablation.csv` | From-scratch ablation |
| `alcohol_baseline.csv` | Single-feature reference |
| `calibration_bins.csv` | Figure 4 |
| `prospective_sensitivity.csv` | Planning scenarios |
| `source_manifest.json` | Dataset repository, resolvable identifier and URL, retrieval note, row counts, features, SHA-256 |

`verification_checks.json` and `final_validation.json` record the checks these outputs passed when they were produced.

---

## Level 2 — Recompute the published values

**Needs:** the raw run directories (~24 GB).

These are not in the repository. Each configuration directory contains `test_predictions_binary.csv`, `test_predictions_ordinal.csv`, `split_indices.npz`, `binary_selection_lock.json`, `ordinal_selection_lock.json`, and `data_audit.json`. Request them from the corresponding author, or regenerate them at Level 3.

With `runs/` and `runs_multiseed/` in place beside this repository:

```bash
python analysis/total_validation.py
```

This rebuilds, from the stored predictions rather than from any summary file:

- Table I sizes against `data_audit.json` and `split_indices.npz`
- Table II AUCs, recomputed from stored probabilities with the recorded clipping
- Table S2 MAE, exact accuracy, within-one accuracy and macro-F1
- Table S3 error counts, with the constant baseline re-derived from development labels
- Equations (4) and (5) on every row
- Table IV AUCs and the Table S1 differences
- Table S4 b/c counts, exact McNemar p-values, and the Holm adjustment across all 19
- Table S5 boundaries and their agreement with Table S4

Then:

```bash
python analysis/resolution_layers.py       # seed resolution, boundary audit, effort-vs-protocol
python analysis/rounding_audit.py          # do printed digits reproduce printed results
python analysis/validate_resolution_claims.py
```

`rounding_audit.py` and `validate_resolution_claims.py` also read the manuscript source, which is not distributed here; they are included because they document the checks that were run.

---

## Level 3 — Rerun the experiments

**Needs:** the datasets, a CUDA GPU, roughly 24 hours.

1. Obtain each source listed in `results/source_manifest.json`. Every entry carries a resolvable identifier (`source_identifier`, `source_url`), a retrieval note, and the SHA-256 of the local file used here; confirm the hash before proceeding.
2. Place them where the loader expects them, and install the training packages in `requirements.txt`.
3. Run the drivers:

```bash
python runner/run_all_datasets.py --mode research --jobs 6
python runner/run_multiseed.py
```

4. Rebuild the audited tables:

```bash
python audit/audit_results.py
python audit/analyze_all_ordinal_baselines.py
python audit/refresh_paper_figures.py
```

Retrained models will not match the recorded ones bit for bit: CUDA execution was not forced to be fully deterministic, and the seed governs candidate sampling, fold assignment, initialization, and the ensemble weight search together. The paper's five-seed sweep quantifies how much that matters — a standard deviation of 0.0208 in red-wine holdout AUC, against paired category differences whose standard deviations are 0.0047 for TabPFN against classical and 0.0142 for from-scratch against classical.

---

## A note on `analysis/legacy/`

Those scripts produced the first aggregation and are kept only for provenance. `aggregate.py` selected each category by its holdout maximum instead of by development rank, which changes reported values. `run_all_analysis.py`, which orchestrates them, lives there for the same reason. Do not use any of them to reproduce the paper. See `analysis/legacy/README.md`.
