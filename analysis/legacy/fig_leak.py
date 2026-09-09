import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
RUNS = Path(r"PAPER_ROOT/runs")
df = pd.read_csv(RUNS/"phase2c_leakage.csv").sort_values("dup_pct")
plt.rcParams.update({"font.size": 9, "figure.dpi": 200})
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
x = np.arange(len(df)); w = .36
ax[0].bar(x-w/2, df.auc_on,  w, label="dedup sebelum split (protokol bersih)", color="#4C78A8")
ax[0].bar(x+w/2, df.auc_off, w, label="tanpa dedup (duplikat menyeberang)",   color="#E45756")
for i, r in enumerate(df.itertuples()):
    ax[0].annotate(f"{r.d_auc:+.3f}", (i, max(r.auc_on, r.auc_off)+.006),
                   ha="center", fontsize=7.5, fontweight="bold")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{d}\n({p:.0f}% dup)" for d, p in zip(df.dataset, df.dup_pct)], fontsize=7)
ax[0].set_ylabel("holdout ROC-AUC"); ax[0].set_ylim(.75, 1.02)
ax[0].set_title("Kebocoran duplikat menggelembungkan skor biner"); ax[0].legend(fontsize=7, loc="lower right")

ax[1].bar(x-w/2, df.exact_on*100,  w, color="#4C78A8")
ax[1].bar(x+w/2, df.exact_off*100, w, color="#E45756")
for i, r in enumerate(df.itertuples()):
    ax[1].annotate(f"{100*r.d_exact:+.1f} pp", (i, max(r.exact_on, r.exact_off)*100+1),
                   ha="center", fontsize=7.5, fontweight="bold")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{d}\n({p:.0f}% dup)" for d, p in zip(df.dataset, df.dup_pct)], fontsize=7)
ax[1].set_ylabel("exact-grade accuracy (%)")
ax[1].set_title("Efeknya jauh lebih besar pada tugas ordinal")
fig.tight_layout(); fig.savefig(RUNS/"figures"/"D_leakage_inflation.png", bbox_inches="tight")
print("tersimpan: figures/D_leakage_inflation.png")
