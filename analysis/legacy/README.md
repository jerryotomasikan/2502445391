# Legacy analysis scripts

These scripts produced the first round of aggregation. They are kept for provenance, not for reuse.

**`aggregate.py` and anything reading `runs/aggregate.csv` are superseded.** That aggregation selected each model category by its *holdout* maximum rather than by the family ranked first on *development*. The difference is not cosmetic: for native red wine it reports a classical AUC of 0.8247 where development selection gives 0.8218.

The published numbers come from the audited path instead:

- `results/audited_results.csv` and `results/category_bootstrap.csv` for category comparisons
- each run's `ordinal_selection_lock.json` for the primary ordinal model, never the first row of `test_metrics_ordinal.csv`, which is sorted by test result

If you are reproducing the paper, use `analysis/` and `audit/`. Use this directory only to trace how the earlier numbers were obtained.
