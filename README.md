# Quantifying Predictive Gains in Tabular Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22679517.svg)](https://doi.org/10.5281/zenodo.22679517)

Analysis and manuscript-generation code for the paper *Quantifying Predictive Gains in Tabular Learning through Ordinal Baselines and Protocol Analysis*.

The paper argues that a reported score should be read against three reference quantities: what a feature-free baseline already achieves, how the evaluation population was assembled, and how far the procedure moves when only its seed changes. This repository contains the code that computes all three, together with the recorded results that back every table in the paper.

## What is here

| Directory | Contents |
|---|---|
| `notebooks/` | `Training.ipynb`, the experiment runner: data loading, deduplication, target construction, development selection, and holdout evaluation for one configuration. |
| `runner/` | Batch drivers that execute the notebook across all configurations and seeds. |
| `analysis/` | The analysis layers reported in the paper: paired error accounting, conditional rejection boundary, seed resolution, and the validators. |
| `analysis/legacy/` | Earlier aggregation scripts, retained for provenance. **Some produce superseded numbers** — see `analysis/legacy/README.md` before using them. |
| `audit/` | Recomputation of every published value from stored predictions, and figure generation. |
| `manuscript/` | Scripts that build the manuscript and validate it against the recorded results. |
| `results/` | The recorded outputs that back every table in the paper. |

## What is not here, and why

**Raw experiment directories.** The complete run artefacts are roughly 24 GB: fitted models, out-of-fold predictions, holdout predictions, split indices, and selection locks for every configuration and seed. They are too large for version control. `results/source_manifest.json` records the source file hashes, and `results/verification_checks.json` records the checks those artefacts passed.

**The datasets themselves.** The sources are public but carry their own terms, and redistributing them here would add nothing that a stable identifier does not. `results/source_manifest.json` gives each source's origin, row counts, feature list, and SHA-256, which is what a reader needs to confirm they have the same file.

**Model checkpoints.** TabPFN weights are distributed by their authors, not by us.

## Reproducing the reported numbers

There are three levels, and they need different things. `docs/REPRODUCE.md` sets out each one.

**Reading the results.** `results/` holds the recorded outputs behind every table, as CSV and JSON. No code is needed to inspect them.

**Recomputing the published values.** `analysis/total_validation.py` rebuilds each table entry from the stored per-run predictions, split indices, and selection locks, then compares the result against the manuscript. It requires the raw run directories, which are not in this repository for size reasons; see `docs/REPRODUCE.md` for how to obtain them.

**Rerunning the experiments.** `notebooks/Training.ipynb` with the drivers in `runner/` reproduces the runs from the datasets. This needs the sources listed in `results/source_manifest.json` and roughly 24 hours of GPU time.

## Environment

Recorded run context: Python 3.12.14, scikit-learn 1.9.0, PyTorch 2.6.0+cu124, XGBoost 3.4.1, CatBoost 1.2.10, tabpfn 8.5.0, on an Intel Core i5-14500 with 64 GB RAM and an NVIDIA RTX 3080. CUDA execution was not forced to be fully deterministic, so retrained models may differ slightly from the recorded ones; the analysis layers operate on the stored predictions and are deterministic.

`requirements.txt` covers the analysis and manuscript stages. Training additionally needs the deep-learning and boosting packages listed there under a separate heading.

## Citation

The archived version used in the paper is [10.5281/zenodo.22679517](https://doi.org/10.5281/zenodo.22679517). The concept DOI [10.5281/zenodo.22679516](https://doi.org/10.5281/zenodo.22679516) always resolves to the latest version.

Machine-readable metadata is in `CITATION.cff`. If you use the reporting structure rather than the code, citing the paper is sufficient.

## Licence

Code is released under the MIT Licence (`LICENSE`). The recorded results in `results/` are released under CC BY 4.0. Neither covers the underlying datasets, which remain under the terms set by their providers.
