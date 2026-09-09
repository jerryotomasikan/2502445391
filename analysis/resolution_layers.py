"""Three analysis layers added on top of the stored predictions.

Nothing here retrains a model or edits an experiment file. Every input is a
recorded artefact: the paired b/c counts already used for the McNemar tests, the
five-seed red-wine records, and the alcohol-only reference AUCs.

Layer 1  resolution of the seeded workflow versus model-family separation
Layer 2  conditional rejection boundary of the exact McNemar design
Layer 3  modelling effort versus a single protocol decision

Outputs land in results/ as CSV plus a summary JSON that the
manuscript builder reads.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
MULTISEED = ROOT / "runs_multiseed"
AUDIT = ROOT / "results"

NODEDUP = {
    "concrete_efreq_nodedup",
    "superconduct_efreq_nodedup",
    "wine_red_nodedup",
    "wine_white_nodedup",
}


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def exact_two_sided_p(b: int, c: int) -> float:
    """Two-sided exact conditional McNemar p-value: Binomial(b + c, 1/2)."""
    d = b + c
    if d == 0:
        return 1.0
    k = max(b, c)
    tail = sum(comb(d, i) for i in range(k, d + 1))
    return min(1.0, 2.0 * tail / (2.0**d))


def rejection_boundary(d: int, alpha: float = 0.05) -> int | None:
    """Smallest |b - c| whose exact two-sided p would not exceed alpha.

    Conditional on d discordant pairs. Returns None when no split of d pairs can
    reach alpha, which is a property of the design rather than of the models.
    """
    for k in range((d // 2) + (d % 2), d + 1):
        if exact_two_sided_p(k, d - k) <= alpha:
            return 2 * k - d
    return None


def label(name: str) -> str:
    out = name.replace("_efreq", " EF").replace("_ewidth", " EW")
    for src, pretty in (
        ("wine_red", "Red wine"),
        ("wine_white", "White wine"),
        ("airquality", "Air quality"),
        ("gasturbine", "Gas turbine"),
        ("superconduct", "Superconductivity"),
        ("california", "California"),
        ("concrete", "Concrete"),
        ("energy", "Energy"),
        ("abalone", "Abalone"),
        ("miami", "Miami"),
    ):
        out = out.replace(src, pretty)
    return out


summary: dict = {}

# ---------------------------------------------------------------- Layer 1 ----
seeds = read(MULTISEED / "multiseed_leakage.csv")
dedup_auc = [float(r["auc_on"]) for r in seeds]
retained_auc = [float(r["auc_off"]) for r in seeds]
protocol_delta = [float(r["d_auc"]) for r in seeds]

# The category differences must come from the audited selection, not from
# runs/aggregate.csv: that earlier aggregation picked each category's holdout
# maximum instead of the family ranked first on development.
audited = read(AUDIT / "audited_results.csv")
principal = [
    r for r in audited
    if r["dataset"] not in NODEDUP and "__seed" not in r["dataset"]
]
if len(principal) != 19:
    raise SystemExit(f"expected 19 principal configurations, found {len(principal)}")

category = {r["dataset"]: r for r in read(AUDIT / "category_bootstrap.csv")}
scratch_gap = [float(category[r["dataset"]]["scratch_delta"]) for r in principal]
pretrained_gap = [float(category[r["dataset"]]["tabpfn_delta"]) for r in principal]
pooled_abs = [abs(x) for x in scratch_gap + pretrained_gap]

# Cross-check the audited category deltas against the stored per-run AUCs.
for record in principal:
    name = record["dataset"]
    scratch_check = float(record["neural_auc"]) - float(record["classic_auc"])
    tabpfn_check = float(record["tabpfn_auc"]) - float(record["classic_auc"])
    for stored, recomputed, tag in (
        (float(category[name]["scratch_delta"]), scratch_check, "scratch"),
        (float(category[name]["tabpfn_delta"]), tabpfn_check, "tabpfn"),
    ):
        if abs(stored - recomputed) > 1e-9:
            raise SystemExit(
                f"category delta mismatch for {name} ({tag}): "
                f"{stored!r} vs {recomputed!r}"
            )

seed_sd = st.stdev(dedup_auc)
pooled_median_abs = st.median(pooled_abs)

summary["seed_resolution"] = {
    "seeds": [r["seed"] for r in seeds],
    "dedup_auc_mean": st.mean(dedup_auc),
    "dedup_auc_sd": seed_sd,
    "dedup_auc_min": min(dedup_auc),
    "dedup_auc_max": max(dedup_auc),
    "dedup_auc_spread": max(dedup_auc) - min(dedup_auc),
    "retained_auc_mean": st.mean(retained_auc),
    "retained_auc_sd": st.stdev(retained_auc),
    "protocol_delta_mean": st.mean(protocol_delta),
    "protocol_delta_sd": st.stdev(protocol_delta),
    "scratch_minus_classical_median": st.median(scratch_gap),
    "scratch_minus_classical_median_abs": st.median([abs(x) for x in scratch_gap]),
    "pretrained_minus_classical_median": st.median(pretrained_gap),
    "pretrained_minus_classical_median_abs": st.median([abs(x) for x in pretrained_gap]),
    "pooled_median_abs": pooled_median_abs,
    "sd_over_pooled_median_abs": seed_sd / pooled_median_abs,
    "protocol_delta_over_seed_sd": st.mean(protocol_delta) / seed_sd,
}

write(
    AUDIT / "seed_resolution.csv",
    ["seed", "auc_dedup", "auc_retained", "protocol_delta"],
    [
        {
            "seed": r["seed"],
            "auc_dedup": f"{float(r['auc_on']):.6f}",
            "auc_retained": f"{float(r['auc_off']):.6f}",
            "protocol_delta": f"{float(r['d_auc']):.6f}",
        }
        for r in seeds
    ],
)

# ---------------------------------------------------------------- Layer 2 ----
inference = {r["dataset"]: r for r in read(AUDIT / "primary_runner_inference.csv")}
rows: list[dict] = []
mismatches: list[str] = []

for record in principal:
    name = record["dataset"]
    source = inference[name]
    b, c = int(source["b"]), int(source["c"])
    d = b + c
    stored_p = float(source["mcnemar_p"])
    recomputed_p = exact_two_sided_p(b, c)
    if abs(stored_p - recomputed_p) > 1e-9:
        mismatches.append(f"{name}: stored {stored_p!r} vs recomputed {recomputed_p!r}")
    boundary = rejection_boundary(d)
    observed = abs(b - c)
    rows.append(
        {
            "task": label(name),
            "dataset": name,
            "discordant": d,
            "observed_margin": observed,
            "required_margin": "" if boundary is None else boundary,
            "attainable": "no" if boundary is None else "yes",
            "reached": "yes" if boundary is not None and observed >= boundary else "no",
            "exact_p": f"{recomputed_p:.4f}",
            "holm_p": f"{float(source['holm_p']):.4f}",
        }
    )

if mismatches:
    raise SystemExit("McNemar p-value cross-check failed:\n  " + "\n  ".join(mismatches))

unattainable = [r for r in rows if r["attainable"] == "no"]
short = [r for r in rows if r["reached"] == "no"]
reached = [r for r in rows if r["reached"] == "yes"]
discordant_counts = [r["discordant"] for r in rows]

summary["rejection_boundary"] = {
    "configurations": len(rows),
    "unattainable": len(unattainable),
    "unattainable_tasks": [r["task"] for r in unattainable],
    "fell_short": len(short),
    "reached_unadjusted": len(reached),
    "reached_tasks": [r["task"] for r in reached],
    "survived_holm": sum(1 for r in rows if float(r["holm_p"]) <= 0.05),
    "discordant_median": st.median(discordant_counts),
    "discordant_min": min(discordant_counts),
    "discordant_max": max(discordant_counts),
    "discordant_total": sum(discordant_counts),
    "mcnemar_crosscheck": "all 19 stored p-values reproduced exactly",
}

write(
    AUDIT / "rejection_boundary.csv",
    ["task", "dataset", "discordant", "observed_margin", "required_margin",
     "attainable", "reached", "exact_p", "holm_p"],
    rows,
)

# ---------------------------------------------------------------- Layer 3 ----
alcohol = {r["dataset"]: float(r["auc_alcohol"]) for r in read(AUDIT / "alcohol_baseline.csv")}
protocol = {r["dataset"]: r for r in read(RUNS / "phase2c_leakage.csv")}
effort_rows: list[dict] = []

for name in ("wine_red", "wine_white"):
    record = next(r for r in audited if r["dataset"] == name)
    primary_auc = float(record["auc"])
    single = alcohol[name]
    modelling = primary_auc - single
    protocol_gap = float(protocol[name]["d_auc"])
    effort_rows.append(
        {
            "source": label(name),
            "dataset": name,
            "auc_single_feature": f"{single:.4f}",
            "auc_primary": f"{primary_auc:.4f}",
            "modelling_gain": f"{modelling:+.4f}",
            "protocol_gain": f"{protocol_gap:+.4f}",
            "protocol_minus_modelling": f"{protocol_gap - modelling:+.4f}",
            "ratio": f"{protocol_gap / modelling:.2f}",
        }
    )

summary["effort_versus_protocol"] = {
    r["dataset"]: {k: v for k, v in r.items() if k != "dataset"} for r in effort_rows
}

write(
    AUDIT / "effort_versus_protocol.csv",
    ["source", "dataset", "auc_single_feature", "auc_primary", "modelling_gain",
     "protocol_gain", "protocol_minus_modelling", "ratio"],
    effort_rows,
)

(AUDIT / "resolution_layers.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

# ------------------------------------------------------------------ report ---
print("Layer 1 — resolution of the seeded workflow")
s = summary["seed_resolution"]
print(f"  deduplicated red-wine AUC over five seeds: mean {s['dedup_auc_mean']:.4f}, "
      f"SD {s['dedup_auc_sd']:.4f}, spread {s['dedup_auc_spread']:.4f}")
print(f"  median |model-family difference| over 19 configurations: {s['pooled_median_abs']:.4f}")
print(f"  seed SD is {s['sd_over_pooled_median_abs']:.2f}x that median")
print(f"  protocol difference is {s['protocol_delta_over_seed_sd']:.2f}x the seed SD, "
      f"and its own SD is only {s['protocol_delta_sd']:.4f}")

print("\nLayer 2 — conditional rejection boundary")
r = summary["rejection_boundary"]
print(f"  {r['fell_short']} of {r['configurations']} configurations fell short of the boundary")
print(f"  {r['unattainable']} could not have reached p<0.05 under any outcome: "
      + ", ".join(r["unattainable_tasks"]))
print(f"  {r['reached_unadjusted']} reached it unadjusted; {r['survived_holm']} survived Holm")
print(f"  discordant pairs: median {r['discordant_median']}, "
      f"range {r['discordant_min']}-{r['discordant_max']}, total {r['discordant_total']}")
print(f"  cross-check: {r['mcnemar_crosscheck']}")

print("\nLayer 3 — modelling effort versus one protocol decision")
for row in effort_rows:
    print(f"  {row['source']}: single feature {row['auc_single_feature']} -> "
          f"primary {row['auc_primary']} ({row['modelling_gain']}); "
          f"protocol {row['protocol_gain']}; ratio {row['ratio']}")
