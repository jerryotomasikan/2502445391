"""Rebuild every published number from the raw run artefacts.

Nothing is taken from the audit CSVs except the list of which run directory each
configuration used. Table values are recomputed from stored predictions, split
indices and selection locks, then compared against the manuscript tables.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
NASKAH = ROOT / "Paper" / "Naskah_Revisi"
AUDIT = NASKAH / "audit"

CLIP = 1e-7
NODEDUP = {"concrete_efreq_nodedup", "superconduct_efreq_nodedup",
           "wine_red_nodedup", "wine_white_nodedup"}

runs = {k: Path(v) for k, v in json.loads((AUDIT / "selected_runs.json").read_text()).items()}
fails: list[str] = []
checks = 0


def ok(cond, msg):
    global checks
    checks += 1
    if not cond:
        fails.append(msg)


def close(a, b, tol=5e-5):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def read(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def md_tables():
    """Parse the manuscript's markdown tables into lists of rows."""
    out, cur = [], []
    for line in (NASKAH / "Paper_Revised.md").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            cur.append(cells)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


TB = md_tables()
NAMES = ["I", "II", "III", "IV", "S1", "S2", "S3", "S4", "S5"]
T = dict(zip(NAMES, TB))


def label_to_ds(label):
    s = label.replace(" EF", "_efreq").replace(" EW", "_ewidth")
    for pretty, src in (("Red wine", "wine_red"), ("White wine", "wine_white"),
                        ("Air quality", "airquality"), ("Gas turbine", "gasturbine"),
                        ("Superconductivity", "superconduct"), ("California", "california"),
                        ("Concrete", "concrete"), ("Energy", "energy"),
                        ("Abalone", "abalone"), ("Miami", "miami")):
        s = s.replace(pretty, src)
    return s


# ---------------------------------------------------------------- helpers ---
def load_run(ds):
    p = runs[ds]
    audit = json.loads((p / "data_audit.json").read_text())
    binlock = json.loads((p / "binary_selection_lock.json").read_text())
    ordlock = json.loads((p / "ordinal_selection_lock.json").read_text())
    binpred = read(p / "test_predictions_binary.csv")
    ordpred = read(p / "test_predictions_ordinal.csv")
    split = np.load(p / "split_indices.npz", allow_pickle=True)
    return p, audit, binlock, ordlock, binpred, ordpred, split


def auc_of(rows, model):
    y = np.array([int(r["binary_true"]) for r in rows])
    p = np.clip(np.array([float(r["p_" + model]) for r in rows]), CLIP, 1 - CLIP)
    return roc_auc_score(y, p)


def dev_constant(audit, ordpred):
    """Re-derive the development-selected constant grade."""
    levels = list(audit["ordinal_levels"])
    clean = Counter({int(k): int(v) for k, v in audit["ordinal_counts_clean"].items()})
    test = Counter(int(r["quality_true"]) for r in ordpred)
    dev = Counter({g: clean.get(g, 0) - test.get(g, 0) for g in levels})
    n_dev = sum(dev.values())
    best, best_cov = None, -1.0
    for c in levels:                      # ties resolved by the smallest grade
        cov = sum(v for g, v in dev.items() if abs(g - c) <= 1) / n_dev
        if cov > best_cov + 1e-12:
            best, best_cov = c, cov
    return best, n_dev


print("=" * 78)
print("A. TABLE I  — sources, sizes, features, test size")
print("=" * 78)
raw_expected = {"Red wine": 1599, "White wine": 4898, "Air quality": 5000,
                "Concrete": 1030, "Energy": 768, "Abalone": 4177,
                "California": 20640, "Miami": 13776, "Gas turbine": 36733,
                "Superconductivity": 21263}
for row in T["I"][1:]:
    name, src_rows, an, feat, testn = row
    ds = label_to_ds(name)
    ds = ds if ds in runs else ds + "_efreq"
    p, audit, *_ , split = load_run(ds)
    ok(int(src_rows.replace(",", "")) == raw_expected[name], f"Table I {name}: source rows")
    ok(int(an.replace(",", "")) == audit["rows_clean"], f"Table I {name}: analyzed n vs data_audit")
    ok(int(feat) == audit["n_features"], f"Table I {name}: feature count")
    ok(int(testn.replace(",", "")) == len(split["test"]), f"Table I {name}: test n vs split_indices")
    ok(audit["rows_clean"] == len(split["development"]) + len(split["test"]),
       f"Table I {name}: dev+test must equal analyzed n")
print(f"  {len(T['I'])-1} sources checked against data_audit.json and split_indices.npz")

print()
print("=" * 78)
print("B. TABLE II — duplicate-handling protocols")
print("=" * 78)
pairs = [("Red wine", "wine_red", "wine_red_nodedup"),
         ("White wine", "wine_white", "wine_white_nodedup"),
         ("Concrete EF", "concrete_efreq", "concrete_efreq_nodedup"),
         ("Superconductivity EF", "superconduct_efreq", "superconduct_efreq_nodedup")]
for label, dedup, retained in pairs:
    row = next(r for r in T["II"][1:] if r[0] == label)
    overlap, a_ded, a_ret, diff = row[1], row[2], row[3], row[4]
    for ds, want_auc in ((dedup, a_ded), (retained, a_ret)):
        p, audit, binlock, ordlock, binpred, ordpred, split = load_run(ds)
        got = auc_of(binpred, binlock["primary_binary_model"])
        ok(close(got, float(want_auc)), f"Table II {label}/{ds}: AUC {got:.6f} vs {want_auc}")
    ok(close(float(a_ret) - float(a_ded), float(diff)),
       f"Table II {label}: difference must equal retained minus dedup")
    # overlap denominator is the retained arm's holdout
    _, _, _, _, _, _, rsplit = load_run(retained)
    ok(int(overlap.split("/")[1]) == len(rsplit["test"]),
       f"Table II {label}: overlap denominator vs retained holdout size")
print(f"  {len(pairs)} protocol pairs: AUCs recomputed from stored probabilities")

print()
print("=" * 78)
print("C. TABLE S2 — ordinal metrics recomputed from predictions")
print("=" * 78)
for row in T["S2"][1:]:
    label, model, K, mae, exact, w1, mf1 = row
    ds = label_to_ds(label)
    p, audit, binlock, ordlock, binpred, ordpred, split = load_run(ds)
    prim = ordlock["primary_ordinal_model"]
    y = np.array([int(r["quality_true"]) for r in ordpred])
    yhat = np.array([int(float(r["pred_" + prim])) for r in ordpred])
    score = np.array([float(r["score_" + prim]) for r in ordpred])
    ok(int(K) == len(audit["ordinal_levels"]), f"S2 {label}: K vs ordinal_levels")
    ok(close(np.abs(score - y).mean(), float(mae)), f"S2 {label}: MAE")
    ok(close(100 * (yhat == y).mean(), float(exact), 5e-3), f"S2 {label}: exact accuracy")
    ok(close(100 * (np.abs(yhat - y) <= 1).mean(), float(w1), 5e-3), f"S2 {label}: within-one")
    got_f1 = f1_score(y, yhat, labels=audit["ordinal_levels"], average="macro", zero_division=0)
    ok(close(got_f1, float(mf1)), f"S2 {label}: macro-F1")
print(f"  {len(T['S2'])-1} configurations: MAE, exact, within-one and macro-F1 all recomputed")

print()
print("=" * 78)
print("D. TABLE S3 — paired error accounting and the constant baseline")
print("=" * 78)
acct = {}
for row in T["S3"][1:]:
    label, n, e0, corr, intro, em, red = row
    ds = label_to_ds(label)
    p, audit, binlock, ordlock, binpred, ordpred, split = load_run(ds)
    prim = ordlock["primary_ordinal_model"]
    y = np.array([int(r["quality_true"]) for r in ordpred])
    yhat = np.array([int(float(r["pred_" + prim])) for r in ordpred])
    const, n_dev = dev_constant(audit, ordpred)
    m_ok = np.abs(yhat - y) <= 1
    c_ok = np.abs(const - y) <= 1
    E0 = int((~c_ok).sum())
    Em = int((~m_ok).sum())
    R = int((~c_ok & m_ok).sum())
    H = int((c_ok & ~m_ok).sum())
    ok(int(n) == len(y), f"S3 {label}: n")
    ok(E0 == int(e0), f"S3 {label}: baseline errors {E0} vs {e0}")
    ok(R == int(corr), f"S3 {label}: corrected {R} vs {corr}")
    ok(H == int(intro), f"S3 {label}: introduced {H} vs {intro}")
    ok(Em == int(em), f"S3 {label}: model errors {Em} vs {em}")
    ok(Em == E0 - R + H, f"S3 {label}: identity Em = E0 - R + H")
    if E0 > 0:
        ok(close(100 * (E0 - Em) / E0, float(red), 5e-3), f"S3 {label}: net reduction")
        # Equation (4) and (5)
        G = (R - H) / len(y)
        A0, Am = 1 - E0 / len(y), 1 - Em / len(y)
        ok(close(G, Am - A0), f"S3 {label}: Eq (4) G = Am - A0")
        ok(close(G, (E0 - Em) / len(y)), f"S3 {label}: Eq (4) G = (E0-Em)/n")
        ok(close((E0 - Em) / E0, G / (1 - A0)), f"S3 {label}: Eq (5) eta = G/(1-A0)")
    else:
        ok(red == "—", f"S3 {label}: dash required when E0 = 0")
    acct[label] = (len(y), E0, Em, const)
print(f"  {len(T['S3'])-1} configurations: constant re-derived from development labels,")
print("  error counts recomputed, identities (4) and (5) verified on every row")

print()
print("=" * 78)
print("E. TABLE III — cross-check against Table S2 and S3")
print("=" * 78)
for row in T["III"][1:]:
    src, ef_m, ef_c, ew_m, ew_c = row
    for suffix, model_pct, const_pct in ((" EF", ef_m, ef_c), (" EW", ew_m, ew_c)):
        label = src + suffix
        n, E0, Em, const = acct[label]
        ok(close(100 * (1 - Em / n), float(model_pct), 5e-3),
           f"Table III {label}: model % vs S3 model errors")
        ok(close(100 * (1 - E0 / n), float(const_pct), 5e-3),
           f"Table III {label}: constant % vs S3 baseline errors")
        s2 = next(r for r in T["S2"][1:] if r[0] == label)
        ok(close(float(s2[5]), float(model_pct), 5e-3),
           f"Table III {label}: model % must equal S2 within-one")
print(f"  {2*(len(T['III'])-1)} cells reconciled with Tables S2 and S3")

print()
print("=" * 78)
print("F. TABLE IV and S1 — category selection and differences")
print("=" * 78)
for row in T["IV"][1:]:
    label, cls, cls_auc, scr, scr_auc, tab_auc = row
    ds = label_to_ds(label)
    p, audit, binlock, ordlock, binpred, ordpred, split = load_run(ds)
    for model_label, want in ((cls, cls_auc), (scr, scr_auc), ("TabPFN", tab_auc)):
        key = {"Extra Trees": "ExtraTrees", "RBF SVM": "SVM_RBF", "HistGB": "HistGradientBoosting",
               "Custom FT": "FTTransformer", "Logistic": "LogisticRegression"}.get(model_label, model_label)
        got = auc_of(binpred, key)
        ok(close(got, float(want)), f"Table IV {label}/{model_label}: AUC {got:.6f} vs {want}")
    s1 = next(r for r in T["S1"][1:] if r[0] == label)
    d_scr = float(re.match(r"([+-][\d.]+)", s1[1]).group(1))
    d_tab = float(re.match(r"([+-][\d.]+)", s1[2]).group(1))
    ok(close(float(scr_auc) - float(cls_auc), d_scr, 1.5e-4),
       f"S1 {label}: scratch delta vs Table IV")
    ok(close(float(tab_auc) - float(cls_auc), d_tab, 1.5e-4),
       f"S1 {label}: TabPFN delta vs Table IV")
print(f"  {3*(len(T['IV'])-1)} AUCs recomputed; {2*(len(T['IV'])-1)} S1 deltas reconciled")

print()
print("=" * 78)
print("G. TABLE S4 and S5 — paired inference")
print("=" * 78)


def exact_p(b, c):
    d = b + c
    if d == 0:
        return 1.0
    k = max(b, c)
    return min(1.0, 2 * sum(comb(d, i) for i in range(k, d + 1)) / 2**d)


def boundary(d, alpha=0.05):
    for k in range((d // 2) + (d % 2), d + 1):
        if exact_p(k, d - k) <= alpha:
            return 2 * k - d
    return None


ps = []
for row in T["S4"][1:]:
    label, n, eff, bc, ep, hp = row
    ds = label_to_ds(label)
    p, audit, binlock, ordlock, binpred, ordpred, split = load_run(ds)
    prim, run = binlock["primary_binary_model"], binlock["runner_up_binary_model"]
    y = np.array([int(r["binary_true"]) for r in binpred])
    a = np.array([int(r["pred_" + prim]) for r in binpred]) == y
    bb = np.array([int(r["pred_" + run]) for r in binpred]) == y
    b, c = int((a & ~bb).sum()), int((~a & bb).sum())
    ok(int(n) == len(y), f"S4 {label}: n")
    ok(f"{b}/{c}" == bc, f"S4 {label}: b/c {b}/{c} vs {bc}")
    ok(close(exact_p(b, c), float(ep)), f"S4 {label}: exact p")
    delta = 100 * (a.mean() - bb.mean())
    ok(close(delta, float(re.match(r"([+-][\d.]+)", eff).group(1)), 5e-3),
       f"S4 {label}: accuracy difference")
    s5 = next(r for r in T["S5"][1:] if r[0] == label)
    ok(int(s5[1]) == b + c, f"S5 {label}: discordant vs S4 b/c")
    ok(int(s5[2]) == abs(b - c), f"S5 {label}: observed margin")
    bd = boundary(b + c)
    ok((s5[3] == "—") if bd is None else (int(s5[3]) == bd), f"S5 {label}: required margin")
    ok(s5[4] == ("yes" if bd is not None and abs(b - c) >= bd else "no"), f"S5 {label}: reached")
    ok(s5[5] == ep and s5[6] == hp, f"S5 {label}: p-values must match S4")
    ps.append((float(ep), label, float(hp)))

order = sorted(ps)
m = len(order)
running = 0.0
for i, (pv, label, holm) in enumerate(order):
    running = max(running, min(1.0, pv * (m - i)))
    ok(close(running, holm, 5e-5), f"Holm {label}: {running:.4f} vs {holm}")
print(f"  {len(T['S4'])-1} comparisons: b/c, exact p, accuracy delta recomputed;")
print("  Holm adjustment recomputed across all 19; S5 reconciled with S4")

print()
print("=" * 78)
print("H. BODY TEXT vs TABLES")
print("=" * 78)
md = (NASKAH / "Paper_Revised.md").read_text(encoding="utf-8")
body = md.split("## Supplementary Results")[0]
abstract = md.split("**Abstract—** ")[1].split("**Index Terms")[0]

n_words = len(abstract.split())
ok(n_words <= 250, f"abstract {n_words} words exceeds 250")

red = next(r for r in T["S3"][1:] if r[0] == "Red wine")
ok(f"from {red[2]} to {red[5]}" in body, "body: red-wine 16 to 9 must match Table S3")
mia = next(r for r in T["S3"][1:] if r[0] == "Miami EW")
ok(f"from {mia[2]} to {mia[5]}" in body, "body: Miami EW 23 to 5 must match Table S3")
ok(f"{float(red[6]):.2f}%" in body, "body: red-wine net reduction")
ok(f"{float(mia[6]):.2f}%" in body, "body: Miami EW net reduction")

pos = sum(1 for r in T["S3"][1:] if r[6] != "—" and float(r[6]) > 0)
ok(pos == 18, f"18 positive-gain configurations expected, found {pos}")
ok("in 18 cases" in body or "in 18 configurations" in body, "body: 18-of-19 claim")

scratch_wins = sum(1 for r in T["IV"][1:] if float(r[4]) > float(r[2]))
tab_wins = sum(1 for r in T["IV"][1:] if float(r[5]) > float(r[2]))
tab_ties = sum(1 for r in T["IV"][1:] if float(r[5]) == float(r[2]))
ok(scratch_wins == 5, f"from-scratch wins {scratch_wins}, body says five")
ok(tab_wins == 15 and tab_ties == 1, f"TabPFN {tab_wins} wins / {tab_ties} ties, body says 15 and one")

gains = []
for row in T["III"][1:]:
    gains.append((float(row[1]) - float(row[2])) - (float(row[3]) - float(row[4])))
ok(all(g > 0 for g in gains), "all seven binning pairs must show a decreased gain")
med = sorted(gains)[len(gains) // 2]
ok(close(med, 25.84, 5e-3), f"median decrease {med:.2f} vs 25.84 in body")

higher = sum(1 for row in T["III"][1:] if float(row[3]) > float(row[1]))
ties = sum(1 for row in T["III"][1:] if float(row[3]) == float(row[1]))
ok(higher == 5 and ties == 1, f"raw EW higher in {higher}, ties {ties}; body says five and energy")

print(f"  abstract {n_words} words; body counts reconciled with Tables III, IV and S3")

print()
print("=" * 78)
print(f"{checks} checks run")
if fails:
    print(f"\n{len(fails)} FAILED")
    for f in fails:
        print("   -", f)
else:
    print("\nEVERY PUBLISHED NUMBER REPRODUCED FROM THE RAW RUN ARTEFACTS")
