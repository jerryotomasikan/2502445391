"""Check every claim the resolution layers introduced against its audit source.

Run after build_manuscript.py. Any drift between the manuscript text and the
recorded artefacts fails loudly rather than silently shipping a wrong number.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NASKAH = ROOT / "Paper" / "Naskah_Revisi"
AUDIT = NASKAH / "audit"

text = (NASKAH / "Paper_Revised.md").read_text(encoding="utf-8")
layers = json.loads((AUDIT / "resolution_layers.json").read_text(encoding="utf-8"))
seed = layers["seed_resolution"]
boundary = layers["rejection_boundary"]
effort = layers["effort_versus_protocol"]

with (AUDIT / "rejection_boundary.csv").open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

failures: list[str] = []
checks = 0


def require(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def present(needle: str, where: str = "manuscript") -> None:
    require(needle in text, f"{where}: missing {needle!r}")


def absent(phrase: str) -> None:
    """Word-boundary search, so 'improves' does not trip a ban on 'proves'."""
    pattern = r"\b" + r"\s+".join(re.escape(w) for w in phrase.split()) + r"\b"
    require(re.search(pattern, text, re.IGNORECASE) is None,
            f"manuscript: unexpected {phrase!r}")


# -- Layer 1 values as written ------------------------------------------------
require(abs(seed["dedup_auc_sd"] - 0.020771) < 5e-7, "seed SD drifted from 0.020771")
present(f"{seed['dedup_auc_sd']:.4f}")
present(f"{seed['dedup_auc_mean']:.4f}")
present(f"{seed['dedup_auc_min']:.4f}")
present(f"{seed['dedup_auc_max']:.4f}")
present(f"{seed['pooled_median_abs']:.4f}")
present(f"{seed['protocol_delta_mean']:.4f}")
present(f"{seed['protocol_delta_sd']:.4f}")

# The prose rounds two ratios; confirm the rounding is the honest one.
require(round(seed["sd_over_pooled_median_abs"]) == 3,
        f"seed SD / median gap is {seed['sd_over_pooled_median_abs']:.2f}, "
        "no longer 'about three times'")
require(abs(seed["protocol_delta_over_seed_sd"] - 3.4) < 0.05,
        f"protocol / seed SD is {seed['protocol_delta_over_seed_sd']:.2f}, not 3.4")
present("3.4 times")
require(abs(seed["pooled_median_abs"] / seed["dedup_auc_sd"] - 1 / 3) < 0.02,
        "median gap is no longer about a third of the seed SD")

# -- Layer 2 counts -----------------------------------------------------------
require(boundary["configurations"] == 19, "boundary audit no longer covers 19 configurations")
require(boundary["fell_short"] == 18, f"fell_short is {boundary['fell_short']}, not 18")
require(boundary["unattainable"] == 6, f"unattainable is {boundary['unattainable']}, not 6")
require(boundary["survived_holm"] == 0, "a comparison now survives Holm adjustment")
require(boundary["discordant_median"] == 16, "median discordant count changed")
require(boundary["discordant_min"] == 2 and boundary["discordant_max"] == 39,
        "discordant range changed")
require(boundary["discordant_total"] == 278, "total discordant pairs changed")
present("18 of 19")
present("median of 16")
present("2 to 39")
present("278 discordant pairs")

named = {"Red wine", "Air quality", "Concrete EF", "Concrete EW", "Energy EF", "Energy EW"}
require(set(boundary["unattainable_tasks"]) == named,
        f"the unattainable set changed: {boundary['unattainable_tasks']}")

# The prose names them as red wine, air quality, both concrete and both energy.
present("native red wine, air quality, and both concrete and both energy variants")

# Every unattainable row must really have at most five discordant pairs, and
# every attainable row must have at least six. That is the claim in the text.
for row in rows:
    d = int(row["discordant"])
    if row["attainable"] == "no":
        require(d <= 5, f"{row['task']}: unattainable with {d} discordant pairs")
    else:
        require(d >= 6, f"{row['task']}: attainable with only {d} discordant pairs")
present("at least 0.0625")

reached = [r for r in rows if r["reached"] == "yes"]
require(len(reached) == 1 and reached[0]["dataset"] == "superconduct_ewidth",
        "the set of configurations reaching the unadjusted boundary changed")

# -- Layer 3 values -----------------------------------------------------------
red, white = effort["wine_red"], effort["wine_white"]
present(red["auc_single_feature"])
present(white["auc_single_feature"])
present(red["modelling_gain"].lstrip("+"))
present(white["modelling_gain"].lstrip("+"))
present(red["protocol_gain"].lstrip("+"))
present(white["protocol_gain"].lstrip("+"))
require(float(red["protocol_gain"]) > float(red["modelling_gain"]),
        "red wine: protocol no longer exceeds modelling, prose must change")
require(float(white["modelling_gain"]) > float(white["protocol_gain"]),
        "white wine: modelling no longer exceeds protocol, prose must change")
present("the protocol difference is the larger one on red wine, the modelling difference on white wine")

# Search-budget wording in the same sentence must match Section III-A.
present("158 classical candidates")
present("three pretrained and sixteen from-scratch configurations")
require("TabPFN adds three configurations" in text
        and "MLP and custom feature-tokenization transformer receive eight configurations each" in text,
        "Section III-A budgets no longer support the 'three pretrained / sixteen from-scratch' wording")

# -- Claims the paper must not make -------------------------------------------
for forbidden in ("proves", "demonstrates equivalence", "statistically equivalent",
                  "causal effect of duplicates", "leakage inflation"):
    absent(forbidden)

# -- Abstract length ----------------------------------------------------------
abstract = text.split("**Abstract—** ")[1].split("**Index Terms")[0]
words = len(abstract.split())
require(words <= 250, f"abstract is {words} words, over the 250 limit")

print(f"{checks} checks run")
if failures:
    print(f"\n{len(failures)} FAILED:")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print(f"abstract {words} words")
print("all resolution-layer claims match their audit sources")
