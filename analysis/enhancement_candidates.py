"""Quantities the manuscript currently asserts without a number.

All computed from the audited, development-selected results. No retraining.
"""
import csv, json, statistics as st
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results"


def read(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


print("=" * 74)
print("E1  Seven binning pairs: direction of the model-minus-constant gain")
print("=" * 74)
rows = read(AUDIT / "ordinal_contrasts.csv")
drops = []
for r in rows:
    ef = float(r["ef_model"]) - float(r["ef_constant"])
    ew = float(r["ew_model"]) - float(r["ew_constant"])
    drops.append((r["source"], ef, ew, ef - ew))
print(f"{'source':18s} {'EF gain':>9s} {'EW gain':>9s} {'decrease':>9s}")
for s, ef, ew, d in drops:
    print(f"{s:18s} {ef:9.2f} {ew:9.2f} {d:9.2f}")

neg = sum(1 for *_, d in drops if d > 0)
n = len(drops)
p_two = min(1.0, 2 * sum(comb(n, k) for k in range(neg, n + 1)) / 2**n)
floor = 2 / 2**n
med = st.median(d for *_, d in drops)
print(f"\ngain fell in {neg} of {n} pairs")
print(f"exact two-sided sign test p = {p_two:.4f}")
print(f"smallest attainable two-sided p with {n} pairs = {floor:.4f}")
print(f"median decrease = {med:.2f} percentage points")

print()
print("=" * 74)
print("E2  How much the selected pair actually disagrees")
print("=" * 74)
inf = read(AUDIT / "primary_runner_inference.csv")
agree, ns, ds = [], [], []
for r in inf:
    n_h, d = int(r["n"]), int(r["discordant"])
    ns.append(n_h)
    ds.append(d)
    agree.append(100 * (1 - d / n_h))
print(f"holdout rows total        : {sum(ns):,}")
print(f"discordant pairs total    : {sum(ds)}")
print(f"pooled discordance        : {100*sum(ds)/sum(ns):.2f}%")
print(f"agreement per task        : median {st.median(agree):.2f}%, "
      f"range {min(agree):.2f}-{max(agree):.2f}%")

print()
print("=" * 74)
print("E3  What holdout size the current disagreement rate would demand")
print("=" * 74)


def boundary(d, alpha=0.05):
    for k in range((d // 2) + (d % 2), d + 1):
        tail = sum(comb(d, i) for i in range(k, d + 1))
        if min(1.0, 2 * tail / 2**d) <= alpha:
            return 2 * k - d
    return None


rate = st.median(d / n_h for d, n_h in zip(ds, ns))
print(f"median discordance rate   : {100*rate:.2f}%")
for target in (6, 12, 20, 40):
    need = boundary(target)
    print(f"  n = {round(target/rate):6,d} gives about {target:2d} discordant pairs; "
          f"a rejection then needs a margin of {need}")

print()
print("=" * 74)
print("E4  Supplementary tables referenced in the body")
print("=" * 74)
md = (ROOT / "manuscript" / "Paper_Revised.md").read_text(encoding="utf-8")
body = md.split("## Supplementary Results")[0]
for t in ("S1", "S2", "S3", "S4", "S5"):
    print(f"  Table {t}: {'referenced' if f'Table {t}' in body else 'NOT referenced'}")
