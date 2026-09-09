"""Fase 2a: MDE McNemar + TOST ekuivalensi + uji berpasangan H1.
Semua dari prediksi holdout tersimpan — tidak ada training ulang."""
import sys, io, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, accuracy_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _common import pick_run

# True = pakai run DENGAN XGBoost/CatBoost/TabPFN (hasil akhir paper).
# False = pakai run tanpa ketiganya (arsip pembanding).
# 1/True (default) = run DENGAN XGBoost/CatBoost/TabPFN; 0/False = tanpa ketiganya.
BOOSTERS = __import__("os").environ.get("PAPER_BOOSTERS", "1") not in ("0", "false", "False")
RUNS = Path(r"PAPER_ROOT/runs")
NEURAL = {"MLP", "FTTransformer"}
PRETRAINED = {"TabPFN"}
CLASSIC = {"LogisticRegression", "SVM_RBF", "RandomForest", "ExtraTrees",
           "HistGradientBoosting", "XGBoost", "CatBoost"}

# Margin ekuivalensi DITETAPKAN DI MUKA (dipilih sebelum melihat hasil TOST;
# bukan preregistrasi formal — harus dinyatakan jujur di paper).
MARGIN_ACC = 0.02     # 2 poin persentase accuracy
MARGIN_AUC = 0.01     # 0.01 ROC-AUC
BOOT = 3000
ALPHA = 0.05


def mde_mcnemar(n_disc, alpha=ALPHA):
    """Selisih menang-kalah minimum yang MASIH bisa signifikan pada n_disc pasangan
    diskordan. Kalau |b-c| di bawah ini, uji tidak akan pernah menolak H0."""
    if n_disc == 0:
        return None
    for k in range(n_disc + 1):
        b = (n_disc + k) // 2
        if (n_disc + k) % 2:
            continue
        if binomtest(min(b, n_disc - b), n=n_disc, p=.5, alternative="two-sided").pvalue < alpha:
            return k
    return None


def boot_deltas(y, pa, pb, ta, tb, seed):
    """Bootstrap berpasangan terstratifikasi kelas: selisih (A - B)."""
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(y == k) for k in (0, 1)]
    d_auc, d_acc = [], []
    for _ in range(BOOT):
        s = np.concatenate([rng.choice(i, size=len(i), replace=True) for i in idx])
        ys = y[s]
        d_auc.append(roc_auc_score(ys, pa[s]) - roc_auc_score(ys, pb[s]))
        d_acc.append(accuracy_score(ys, pa[s] >= ta) - accuracy_score(ys, pb[s] >= tb))
    return np.asarray(d_auc), np.asarray(d_acc)


def tost(deltas, margin, alpha=ALPHA):
    """Ekuivalensi via CI (1-2*alpha) dari distribusi bootstrap."""
    lo, hi = np.quantile(deltas, [alpha, 1 - alpha])          # CI 90% untuk alpha=.05
    return bool(lo > -margin and hi < margin), lo, hi


rows = []
for d in sorted(p for p in RUNS.iterdir() if p.is_dir() and p.name != "figures"):
    rd = pick_run(d, boosters=BOOSTERS)
    if rd is None or not (rd / "test_predictions_binary.csv").is_file():
        continue
    pred = pd.read_csv(rd / "test_predictions_binary.csv")
    met = pd.read_csv(rd / "test_metrics_binary.csv").set_index("model")
    stat = json.loads((rd / "statistical_summary.json").read_text(encoding="utf-8"))
    y = pred["binary_true"].to_numpy()
    P = lambda m: pred[f"p_{m}"].to_numpy()
    T = lambda m: float(met.loc[m, "threshold"])

    avail = set(met.index)
    best_c = max(CLASSIC & avail, key=lambda m: met.loc[m, "roc_auc"])
    best_n = max(NEURAL & avail, key=lambda m: met.loc[m, "roc_auc"], default=None)
    best_p = max(PRETRAINED & avail, key=lambda m: met.loc[m, "roc_auc"], default=None)
    prim, run = stat["primary"], stat["runner_up"]
    nb, nc = stat["mcnemar_b"], stat["mcnemar_c"]
    ndisc = nb + nc

    # --- H2: primary vs runner-up (klaim ensemble) ---
    a_auc, a_acc = boot_deltas(y, P(prim), P(run), T(prim), T(run), 20260906)
    eq_acc, lo_acc, hi_acc = tost(a_acc, MARGIN_ACC)
    # --- H1: neural terbaik vs klasik terbaik ---
    h_auc, _ = boot_deltas(y, P(best_n), P(best_c), T(best_n), T(best_c), 20260907)
    eq_auc, lo_auc, hi_auc = tost(h_auc, MARGIN_AUC)
    ci_auc = np.quantile(h_auc, [.025, .975])
    # --- pralatih (TabPFN) vs klasik terbaik ---
    if best_p is not None:
        p_auc, _ = boot_deltas(y, P(best_p), P(best_c), T(best_p), T(best_c), 20260908)
        ci_p = np.quantile(p_auc, [.025, .975])
        d_p = met.loc[best_p, "roc_auc"] - met.loc[best_c, "roc_auc"]
    else:
        ci_p, d_p = (np.nan, np.nan), np.nan

    rows.append(dict(
        dataset=d.name, n_hold=len(pred), n_disc=ndisc,
        mde_menang=mde_mcnemar(ndisc), mcn_selisih=abs(nb - nc), mcn_p=stat["mcnemar_exact_p"],
        h2_d_acc=a_acc.mean(), h2_lo90=lo_acc, h2_hi90=hi_acc, h2_ekuiv=eq_acc,
        h1_model=f"{best_n} vs {best_c}",
        h1_d_auc=met.loc[best_n, "roc_auc"] - met.loc[best_c, "roc_auc"],
        h1_ci_lo=ci_auc[0], h1_ci_hi=ci_auc[1],
        h1_p_neural_unggul=float((h_auc > 0).mean()),
        h1_lo90=lo_auc, h1_hi90=hi_auc, h1_ekuiv=eq_auc,
        pre_model=best_p, pre_d_auc=d_p, pre_ci_lo=ci_p[0], pre_ci_hi=ci_p[1],
    ))

df = pd.DataFrame(rows)
df.to_csv(RUNS / "phase2_statistics.csv", index=False)
print(f"tersimpan: {RUNS/'phase2_statistics.csv'}\n")

print("=" * 108)
print(f"A. TENAGA UJI McNEMAR — 'mde_menang' = selisih menang-kalah MINIMUM agar p<0.05 bisa tercapai")
print("=" * 108)
t = df[["dataset", "n_hold", "n_disc", "mcn_selisih", "mde_menang", "mcn_p"]].copy()
t["cukup_tenaga"] = np.where(t.mde_menang.isna(), "TIDAK (0 diskordan)",
                             np.where(t.mcn_selisih >= t.mde_menang.fillna(1e9), "ya", "tidak"))
print(t.to_string(index=False))
print(f"\n  Dataset yang secara struktural TIDAK MUNGKIN mencapai p<0.05: "
      f"{(t.mde_menang.isna() | (t.mde_menang > t.n_disc)).sum()}/19")
