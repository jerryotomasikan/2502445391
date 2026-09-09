from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
"""Analisis berpasangan yang benar: kontras efreq vs ewidth pada dataset SAMA."""
import sys, io
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest, spearmanr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
df = pd.read_csv(str(REPO_ROOT / r"runs/aggregate.csv")).set_index("dataset")

BASES = ["concrete", "energy", "california", "abalone", "gasturbine", "superconduct", "miami"]
rows = []
for b in BASES:
    e, w = df.loc[f"{b}_efreq"], df.loc[f"{b}_ewidth"]
    rows.append(dict(base=b, imb_e=e.imb, imb_w=w.imb,
                     gain_e=e.w1_gain, gain_w=w.w1_gain, d_gain=w.w1_gain - e.w1_gain,
                     mf1_e=e.mf1, mf1_w=w.mf1, d_mf1=w.mf1 - e.mf1,
                     w1_e=e.w1, w1_w=w.w1))
p = pd.DataFrame(rows)

print("=" * 100)
print("KONTRAS BERPASANGAN — dataset sama, hanya skema binning berbeda")
print("  efreq = 6 bin frekuensi-sama (seimbang) | ewidth = 6 bin lebar-sama (timpang)")
print("=" * 100)
print(f"{'dataset':14s} {'imb efreq':>10s} {'imb ewidth':>11s} {'gain efreq':>11s} {'gain ewidth':>12s} {'Δgain':>8s} {'ΔmacroF1':>9s}")
for _, r in p.iterrows():
    print(f"{r.base:14s} {r.imb_e:10.1f} {r.imb_w:11.1f} {r.gain_e:+11.3f} {r.gain_w:+12.3f} "
          f"{r.d_gain:+8.3f} {r.d_mf1:+9.3f}")

n = len(p)
turun = int((p.d_gain < 0).sum())
sign_p = binomtest(turun, n=n, p=.5, alternative="two-sided").pvalue
w_stat, w_p = wilcoxon(p.gain_w, p.gain_e)
print()
print(f"  Ketimpangan NAIK di lengan ewidth pada  : {int((p.imb_w > p.imb_e).sum())}/{n} pasangan")
print(f"  Informativitas within-one TURUN pada    : {turun}/{n} pasangan")
print(f"  Uji tanda (binomial dua sisi)           : p = {sign_p:.4f}")
print(f"  Wilcoxon signed-rank                    : W = {w_stat:.1f}, p = {w_p:.4f}")
print(f"  Median Δgain (ewidth - efreq)           : {p.d_gain.median():+.3f}")
print()
print("=" * 100)
print("PEMBANDING: korelasi lintas 19 konfigurasi (DESKRIPTIF SAJA — tidak bebas)")
print("=" * 100)
r19, p19 = spearmanr(df.imb, df.w1_gain)
base10 = df.loc[[i for i in df.index if not i.endswith("_ewidth")]]
r10, p10 = spearmanr(base10.imb, base10.w1_gain)
print(f"  seluruh 19 konfigurasi           : rho = {r19:+.3f}, p = {p19:.2e}  <- PSEUDO-REPLIKASI")
print(f"  hanya 1 konfigurasi per dataset  : rho = {r10:+.3f}, p = {p10:.4f}  (n = {len(base10)})")
print()
print("  KESIMPULAN METODOLOGIS: laporkan uji tanda berpasangan (n=7) sebagai hasil")
print("  inferensial; korelasi lintas-konfigurasi hanya sebagai ilustrasi deskriptif,")
print("  dengan ketidakbebasan dinyatakan eksplisit.")
p.to_csv(str(REPO_ROOT / r"runs/paired_binning_contrast.csv"), index=False)
