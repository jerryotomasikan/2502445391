"""Fase 1: agregasi 19 run -> tabel induk + uji apakah klaim paper berdiri."""
import json, glob, sys, io
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _common import pick_run

# True = pakai run DENGAN XGBoost/CatBoost/TabPFN (hasil akhir paper).
# False = pakai run tanpa ketiganya (arsip pembanding).
# 1/True (default) = run DENGAN XGBoost/CatBoost/TabPFN; 0/False = tanpa ketiganya.
BOOSTERS = __import__("os").environ.get("PAPER_BOOSTERS", "1") not in ("0", "false", "False")
RUNS = REPO_ROOT / r"runs"
NEURAL = {"MLP", "FTTransformer"}          # dilatih dari nol
PRETRAINED = {"TabPFN"}                    # transformer pralatih [5]
CLASSIC = {"LogisticRegression", "SVM_RBF", "RandomForest", "ExtraTrees", "HistGradientBoosting",
           "XGBoost", "CatBoost"}

rows = []
for d in sorted(RUNS.iterdir()):
    if not d.is_dir():
        continue
    rd = pick_run(d, boosters=BOOSTERS, verbose=True)
    if rd is None or not (rd / "test_metrics_binary.csv").is_file():
        continue
    audit = json.loads((rd / "data_audit.json").read_text(encoding="utf-8"))
    stat = json.loads((rd / "statistical_summary.json").read_text(encoding="utf-8"))
    tb = pd.read_csv(rd / "test_metrics_binary.csv")
    to = pd.read_csv(rd / "test_metrics_ordinal.csv")
    bs = pd.read_csv(rd / "paired_bootstrap_primary_vs_runner.csv")
    op = pd.read_csv(rd / "test_predictions_ordinal.csv")

    def auc_of(names):
        s = tb[tb.model.isin(names)]
        return (s.roc_auc.max(), s.loc[s.roc_auc.idxmax(), "model"]) if len(s) else (np.nan, "-")

    cls_auc, cls_name = auc_of(CLASSIC)
    neu_auc, neu_name = auc_of(NEURAL)
    pre_auc, pre_name = auc_of(PRETRAINED)
    sv = tb[tb.model == "SoftVoting"]
    sv_auc = float(sv.roc_auc.iloc[0]) if len(sv) else np.nan

    # ordinal: model utama (baris pertama = MAE terendah) + baseline "tebak kelas modus"
    prim_ord = to.iloc[0]
    y = op["quality_true"].to_numpy()
    lv, cnt = np.unique(y, return_counts=True)
    modal = int(lv[cnt.argmax()])
    trivial_w1 = float(np.mean(np.abs(y - modal) <= 1))     # tebak modus utk semua
    counts = np.array(audit["ordinal_counts_clean"].values() if isinstance(audit["ordinal_counts_clean"], dict) else [])
    c = np.array(list(audit["ordinal_counts_clean"].values()), dtype=float)
    imbalance = float(c.max() / c.min())                     # rasio kelas terbesar:terkecil

    rows.append(dict(
        dataset=d.name, n=audit["rows_clean"], p=audit["n_features"],
        n_hold=len(op), dup=audit["duplicates_removed"], konflik=audit["conflicting_feature_groups"],
        lvl=len(audit["ordinal_levels"]), imb=imbalance,
        auc_klasik=cls_auc, klasik=cls_name, auc_neural=neu_auc, neural=neu_name,
        d_neural=neu_auc - cls_auc, auc_sv=sv_auc,
        auc_pretrained=pre_auc, pretrained=pre_name,
        d_pretrained=pre_auc - cls_auc,
        primary=stat["primary"], runner=stat["runner_up"],
        mcn_b=stat["mcnemar_b"], mcn_c=stat["mcnemar_c"], mcn_p=stat["mcnemar_exact_p"],
        d_acc=float(bs.loc[bs.metric == "Accuracy", "delta_primary_minus_runner"].iloc[0]),
        acc_lo=float(bs.loc[bs.metric == "Accuracy", "ci95_low"].iloc[0]),
        acc_hi=float(bs.loc[bs.metric == "Accuracy", "ci95_high"].iloc[0]),
        ord_model=prim_ord["model"], mae=prim_ord["mae"], qwk=prim_ord["qwk"],
        exact=prim_ord["exact_accuracy"], w1=prim_ord["within_one_accuracy"],
        mf1=prim_ord["macro_f1"], w1_trivial=trivial_w1, w1_gain=prim_ord["within_one_accuracy"] - trivial_w1,
    ))

df = pd.DataFrame(rows)
df.to_csv(RUNS / "aggregate.csv", index=False)
print(f"tersimpan: {RUNS / 'aggregate.csv'}  ({len(df)} baris)\n")

print("=" * 118)
print("TABEL INDUK — BINER (holdout)")
print("=" * 118)
b = df[["dataset", "n", "p", "n_hold", "auc_klasik", "auc_neural", "d_neural", "auc_sv",
        "primary", "runner", "mcn_b", "mcn_c", "mcn_p"]].copy()
for col in ["auc_klasik", "auc_neural", "d_neural", "auc_sv"]:
    b[col] = b[col].round(4)
b["mcn_p"] = b["mcn_p"].round(3)
print(b.to_string(index=False))
