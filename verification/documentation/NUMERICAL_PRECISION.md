# Numeric precision in verification

Original prediction and metadata files are unchanged. CSVs are read with float_precision="round_trip" so probabilities at a threshold retain their decimal representation. Binary predictions and thresholds are checked without changing them.

Original TabPFN/XGBoost float32 ordinal scores were exported with finite decimal precision. Recomputed MAE differs from original in-memory MAE by at most about 1.15e-8 grade units. MAE verification therefore uses absolute tolerance 5e-8; other metrics and interval checks use 1e-9. This is far below four reported decimal places and does not change any published value.

The original published-table checker had compared two rounded AUCs with a separately rounded difference, and had multiplied a rounded McNemar p-value for Holm adjustment. The portable numeric checker correctly uses full-precision AUCs and exact McNemar probabilities before comparing to published display precision. This changes validation arithmetic only; no predictions, model choices, data splits or experimental metrics were changed.

Bootstrap verification preserves the actual random stream of each original calculation: sequential class draws for primary/runner-up, batches of 100 draws for category AUC comparisons, and class-wise arrays of 3,000 draws for the two ordinal-baseline examples. Reusing the same numeric seed with a different draw order does not reproduce the same Monte Carlo sample.
