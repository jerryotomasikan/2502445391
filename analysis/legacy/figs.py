import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
from scipy.stats import spearmanr

RUNS = REPO_ROOT / r"runs"
df = pd.read_csv(RUNS / "aggregate.csv")
df["ceiling"] = df[["auc_klasik", "auc_neural", "auc_sv"]].max(axis=1) >= 0.97
OUT = RUNS / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200})

# --- Gambar A: selisih neural - klasik per dataset ---
s = df.sort_values("d_neural")
fig, ax = plt.subplots(figsize=(7.2, 5.0))
col = ["#BBBBBB" if c else ("#54A24B" if d > 0 else "#E45756") for c, d in zip(s.ceiling, s.d_neural)]
ax.barh(s.dataset, s.d_neural, color=col, edgecolor="white")
ax.axvline(0, color="black", lw=.9)
ax.set_xlabel("holdout ROC-AUC:  neural terbaik  −  klasik terbaik")
ax.set_title("Model neural tidak konsisten mengungguli klasik (19 konfigurasi)")
h = [plt.Rectangle((0,0),1,1,color=c) for c in ["#54A24B", "#E45756", "#BBBBBB"]]
ax.legend(h, ["neural unggul", "klasik unggul", "efek langit-langit (AUC≥0.97)"], fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "A_neural_minus_classical.png", bbox_inches="tight"); plt.close(fig)

# --- Gambar B: informativitas within-one vs ketimpangan ---
r, p = spearmanr(df.imb, df.w1_gain)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.axhline(0, color="#E45756", ls="--", lw=1.2)
ax.scatter(df.imb, df.w1_gain, s=46, c=np.where(df.dataset.str.contains("wine"), "#B279A2", "#4C78A8"),
           edgecolor="white", zorder=3)
for _, row in df.iterrows():
    if row.w1_gain < .15 or row.imb > 40 or "wine" in row.dataset:
        ax.annotate(row.dataset, (row.imb, row.w1_gain), fontsize=6,
                    xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("ketimpangan level ordinal  (kelas terbesar / terkecil, skala log)")
ax.set_ylabel("within-one accuracy  −  baseline tebak-kelas-modus")
ax.set_title(f"Makin timpang, within-one accuracy makin tak informatif\nSpearman ρ = {r:+.3f}  (p = {p:.1e},  n = 19)")
ax.text(.02, .04, "di bawah garis merah: model KALAH dari tebakan konstan",
        transform=ax.transAxes, fontsize=7, color="#E45756")
fig.tight_layout(); fig.savefig(OUT / "B_within_one_illusion.png", bbox_inches="tight"); plt.close(fig)

# --- Gambar C: within-one vs exact, ukuran = ketimpangan ---
fig, ax = plt.subplots(figsize=(6.4, 4.8))
sc = ax.scatter(df.exact, df.w1, s=24 + 42 * np.log10(df.imb + 1), c=df.mf1,
                cmap="viridis", edgecolor="white", zorder=3)
ax.plot([0, 1], [0, 1], color="#999999", ls=":", lw=1)
for _, row in df.iterrows():
    if row.mf1 < .45 or "wine" in row.dataset:
        ax.annotate(row.dataset, (row.exact, row.w1), fontsize=6, xytext=(4, -8), textcoords="offset points")
ax.set_xlabel("exact-grade accuracy"); ax.set_ylabel("within-one accuracy")
ax.set_title("Within-one tinggi menyembunyikan exact rendah\n(ukuran titik = ketimpangan, warna = macro-F1)")
fig.colorbar(sc, ax=ax, label="macro-F1")
fig.tight_layout(); fig.savefig(OUT / "C_within_one_vs_exact.png", bbox_inches="tight"); plt.close(fig)

print("tersimpan di", OUT)
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:34s} {f.stat().st_size/1024:6.0f} KB")
