import sys, io
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
df = pd.read_csv(r"PAPER_ROOT/runs\aggregate.csv")

print("=" * 100)
print("PERIKSA EFEK LANGIT-LANGIT (ceiling) — kalau AUC ~1.0, 'neural tak menang' jadi trivial")
print("=" * 100)
df["ceiling"] = df[["auc_klasik", "auc_neural", "auc_sv"]].max(axis=1) >= 0.97
c, nc = df[df.ceiling], df[~df.ceiling]
print(f"  dataset LANGIT-LANGIT (AUC>=0.97): {len(c)}/19 -> {sorted(c.dataset)}")
print(f"  dataset INFORMATIF  (AUC< 0.97): {len(nc)}/19 -> {sorted(nc.dataset)}")
print()
for label, s in [("SEMUA 19", df), ("HANYA INFORMATIF", nc)]:
    win = (s.d_neural > 0).sum()
    print(f"  {label:18s}: neural menang {win}/{len(s)} | median Δ = {s.d_neural.median():+.4f} "
          f"| McNemar p<0.05: {(s.mcn_p < .05).sum()}/{len(s)} | diskordan b+c median = {(s.mcn_b + s.mcn_c).median():.0f}")

print()
print("  Pasangan diskordan McNemar per dataset (tenaga uji):")
t = df[["dataset", "mcn_b", "mcn_c", "mcn_p"]].copy()
t["b_plus_c"] = t.mcn_b + t.mcn_c
print(t.sort_values("b_plus_c")[["dataset", "b_plus_c", "mcn_p"]].to_string(index=False))

print()
print("=" * 100)
print("HUBUNGAN KETIMPANGAN -> INFORMATIVITAS within-one  (w1_gain = w1 - w1_trivial)")
print("=" * 100)
r, p = spearmanr(df.imb, df.w1_gain)
print(f"  Spearman(ketimpangan, w1_gain) = {r:+.3f}   p = {p:.4f}   (n=19)")
r2, p2 = spearmanr(df.imb, df.mf1)
print(f"  Spearman(ketimpangan, macro-F1) = {r2:+.3f}  p = {p2:.4f}")
print()
print("  Kasus paling telanjang (w1 tinggi tapi nyaris = tebakan konstan):")
worst = df.nsmallest(5, "w1_gain")[["dataset", "imb", "w1", "w1_trivial", "w1_gain", "exact", "mf1"]]
print(worst.round(3).to_string(index=False))
