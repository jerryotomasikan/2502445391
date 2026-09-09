import matplotlib; matplotlib.use("Agg")
import sys, io
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
R = REPO_ROOT / r"runs"
ms = pd.read_csv(R.parent / "runs_multiseed" / "multiseed_leakage.csv")
ag = pd.read_csv(R / "aggregate.csv")

sd_seed = max(ms.auc_on.std(ddof=1), ms.auc_off.std(ddof=1))
med_model = ag.d_neural.abs().median()
med_pre = ag.d_pretrained.abs().median()
print("=" * 92)
print("TEMUAN TAMBAHAN — derau seed vs selisih antar-model")
print("=" * 92)
print(f"  SD ROC-AUC hanya karena mengganti seed pembagian : {sd_seed:.4f}")
print(f"  |Δ| median neural-dari-nol vs klasik             : {med_model:.4f}")
print(f"  |Δ| median pralatih vs klasik                    : {med_pre:.4f}")
print(f"  efek kebocoran (rata-rata 5 seed)                : {ms.d_auc.mean():.4f} ± {ms.d_auc.std(ddof=1):.4f}")
print()
print(f"  -> Mengganti seed saja menggeser skor {sd_seed/med_model:.1f}x lebih besar")
print(f"     daripada selisih median antar-keluarga model.")
print(f"  -> Efek kebocoran {ms.d_auc.mean()/sd_seed:.1f}x derau seed: bertahan, tapi harus")
print(f"     dilaporkan sebagai rentang, bukan satu angka.")

fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
x = np.arange(len(ms))
ax.plot(x, ms.auc_on, "o-", color="#4C78A8", label="dedup sebelum split")
ax.plot(x, ms.auc_off, "s-", color="#E45756", label="tanpa dedup")
for i, r in enumerate(ms.itertuples()):
    ax.annotate(f"+{r.d_auc:.3f}", (i, (r.auc_on + r.auc_off) / 2),
                ha="center", fontsize=7.5, color="#333")
    ax.vlines(i, r.auc_on, r.auc_off, color="#BBB", lw=1, zorder=0)
ax.set_xticks(x); ax.set_xticklabels([str(int(s)) for s in ms.seed], fontsize=8)
ax.set(xlabel="seed pembagian data", ylabel="holdout ROC-AUC (model utama)",
       title=f"Inflasi kebocoran bertahan di kelima seed\n"
             f"rata-rata {ms.d_auc.mean():+.4f} ± {ms.d_auc.std(ddof=1):.4f}  "
             f"({ms.d_auc.mean()/sd_seed:.1f}x derau antar-seed)")
ax.legend(fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(R / "figures" / "F_multiseed_leakage.png", bbox_inches="tight")
print(f"\n  tersimpan: runs/figures/F_multiseed_leakage.png")
