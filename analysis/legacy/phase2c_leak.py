"""Fase 2c: inflasi akibat kebocoran duplikat (dedup ON vs OFF)."""
import sys, io, json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _common import pick_run

# 1/True (default) = run DENGAN XGBoost/CatBoost/TabPFN; 0/False = tanpa ketiganya.
BOOSTERS = __import__("os").environ.get("PAPER_BOOSTERS", "1") not in ("0", "false", "False")
RUNS = REPO_ROOT / r"runs"
CLASSIC = {"LogisticRegression","SVM_RBF","RandomForest","ExtraTrees","HistGradientBoosting"}
NEURAL  = {"MLP","FTTransformer"}

def load(name):
    d = RUNS / name
    rd = pick_run(d, boosters=BOOSTERS)
    if rd is None:
        raise FileNotFoundError(f"tidak ada run selesai (boosters={BOOSTERS}) untuk {name}")
    return (json.loads((rd/"data_audit.json").read_text(encoding="utf-8")),
            pd.read_csv(rd/"test_metrics_binary.csv").set_index("model"),
            pd.read_csv(rd/"test_metrics_ordinal.csv"),
            json.loads((rd/"statistical_summary.json").read_text(encoding="utf-8")))

PAIRS = [("wine_red","wine_red_nodedup"), ("wine_white","wine_white_nodedup"),
         ("concrete_efreq","concrete_efreq_nodedup"),
         ("superconduct_efreq","superconduct_efreq_nodedup")]

rows = []
for on, off in PAIRS:
    a_ad, a_b, a_o, a_s = load(on)     # dedup ON  (protokol bersih)
    b_ad, b_b, b_o, b_s = load(off)    # dedup OFF (duplikat menyeberang)
    pct_dup = 100 * b_ad["duplicates_removed"] / b_ad["rows_raw"]
    bc_on  = max(CLASSIC & set(a_b.index), key=lambda m: a_b.loc[m,"roc_auc"])
    bc_off = max(CLASSIC & set(b_b.index), key=lambda m: b_b.loc[m,"roc_auc"])
    rows.append(dict(
        dataset=on, dup_pct=pct_dup, n_on=a_ad["rows_clean"], n_off=b_ad["rows_clean"],
        auc_on=a_b.loc[a_s["primary"],"roc_auc"], auc_off=b_b.loc[b_s["primary"],"roc_auc"],
        acc_on=a_b.loc[a_s["primary"],"accuracy"], acc_off=b_b.loc[b_s["primary"],"accuracy"],
        klasik_on=a_b.loc[bc_on,"roc_auc"], klasik_off=b_b.loc[bc_off,"roc_auc"],
        mae_on=a_o.iloc[0]["mae"], mae_off=b_o.iloc[0]["mae"],
        exact_on=a_o.iloc[0]["exact_accuracy"], exact_off=b_o.iloc[0]["exact_accuracy"],
    ))

df = pd.DataFrame(rows)
df["d_auc"]   = df.auc_off   - df.auc_on
df["d_acc"]   = df.acc_off   - df.acc_on
df["d_klasik"]= df.klasik_off- df.klasik_on
df["d_exact"] = df.exact_off - df.exact_on
df["d_mae"]   = df.mae_off   - df.mae_on
df.to_csv(RUNS/"phase2c_leakage.csv", index=False)

print("="*104)
print("INFLASI KEBOCORAN DUPLIKAT  (dedup OFF  minus  dedup ON).  Positif = skor menggelembung")
print("="*104)
print(f"{'dataset':20s} {'%dup':>6s} {'n(on/off)':>13s} {'AUC on':>8s} {'AUC off':>8s} {'ΔAUC':>8s} "
      f"{'acc on':>7s} {'acc off':>8s} {'Δacc(pp)':>9s}")
for _, r in df.iterrows():
    print(f"{r.dataset:20s} {r.dup_pct:5.1f}% {r.n_on:6,}/{r.n_off:<6,} "
          f"{r.auc_on:8.4f} {r.auc_off:8.4f} {r.d_auc:+8.4f} "
          f"{r.acc_on:7.4f} {r.acc_off:8.4f} {100*r.d_acc:+9.2f}")
print()
print(f"  Median inflasi AUC  (model utama) : {df.d_auc.median():+.4f}")
print(f"  Median inflasi AUC  (klasik terbaik): {df.d_klasik.median():+.4f}")
print(f"  Median inflasi accuracy            : {100*df.d_acc.median():+.2f} poin persentase")
print()
print("="*104)
print("EFEK PADA TUGAS ORDINAL")
print("="*104)
print(f"{'dataset':20s} {'MAE on':>8s} {'MAE off':>8s} {'ΔMAE':>8s} {'exact on':>9s} {'exact off':>10s} {'Δexact(pp)':>11s}")
for _, r in df.iterrows():
    print(f"{r.dataset:20s} {r.mae_on:8.4f} {r.mae_off:8.4f} {r.d_mae:+8.4f} "
          f"{r.exact_on:9.4f} {r.exact_off:10.4f} {100*r.d_exact:+11.2f}")
print()
print(f"  Median ΔMAE   : {df.d_mae.median():+.4f}  (negatif = error tampak lebih kecil dari seharusnya)")
print(f"  Median Δexact : {100*df.d_exact.median():+.2f} poin persentase")
print()
print("  CATATAN: kedua lengan memakai holdout BERBEDA (dedup mengubah data), jadi ini")
print("  bukan uji berpasangan melainkan perbandingan dua protokol evaluasi. Itu memang")
print("  pertanyaannya: berapa besar skor menggelembung kalau duplikat dibiarkan menyeberang.")
