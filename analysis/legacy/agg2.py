import sys, io
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
df = pd.read_csv(r"PAPER_ROOT/runs\aggregate.csv")

print("=" * 112)
print("TABEL INDUK — ORDINAL (holdout).  w1_trivial = within-one kalau SEMUA ditebak kelas modus")
print("=" * 112)
o = df[["dataset", "lvl", "imb", "ord_model", "mae", "qwk", "exact", "w1", "w1_trivial", "w1_gain", "mf1"]].copy()
o = o.sort_values("imb")
for c in ["imb", "mae", "qwk", "exact", "w1", "w1_trivial", "w1_gain", "mf1"]:
    o[c] = o[c].round(4)
print(o.to_string(index=False))

print()
print("=" * 112)
print("UJI KLAIM 1 — apakah neural konsisten mengungguli klasik?")
print("=" * 112)
menang = df[df.d_neural > 0]
print(f"  neural > klasik : {len(menang)}/19 dataset")
print(f"  margin saat menang: {menang.d_neural.round(4).tolist()}")
print(f"  klasik >= neural : {19-len(menang)}/19, margin terburuk neural {df.d_neural.min():.4f} ({df.loc[df.d_neural.idxmin(),'dataset']})")
print(f"  median Δ(neural-klasik) = {df.d_neural.median():+.4f}   rata-rata = {df.d_neural.mean():+.4f}")

print()
print("=" * 112)
print("UJI KLAIM 2 — apakah SoftVoting mengungguli konstituen terbaiknya?")
print("=" * 112)
print(f"  McNemar p minimum lintas 19 dataset : {df.mcn_p.min():.3f} ({df.loc[df.mcn_p.idxmin(),'dataset']})")
print(f"  jumlah dataset dengan p < 0.05      : {(df.mcn_p < .05).sum()}/19")
print(f"  CI bootstrap accuracy memuat nol    : {((df.acc_lo <= 0) & (df.acc_hi >= 0)).sum()}/19")

print()
print("=" * 112)
print("UJI KLAIM 3 — ilusi metrik toleransi, DIKONTROL oleh keseimbangan (efreq vs ewidth)")
print("=" * 112)
print(f"{'pasangan':16s} {'skema':8s} {'imb':>7s} {'exact':>7s} {'w1':>7s} {'w1_trivial':>11s} {'w1_gain':>8s} {'macroF1':>8s}")
for base in ["concrete", "energy", "california", "abalone", "gasturbine", "superconduct", "miami"]:
    for suf in ["efreq", "ewidth"]:
        r = df[df.dataset == f"{base}_{suf}"]
        if len(r):
            r = r.iloc[0]
            print(f"{base:16s} {suf:8s} {r.imb:7.1f} {r.exact:7.3f} {r.w1:7.3f} {r.w1_trivial:11.3f} {r.w1_gain:+8.3f} {r.mf1:8.3f}")
    print()

e = df[df.dataset.str.endswith("_efreq")]
w = df[df.dataset.str.endswith("_ewidth")]
print(f"  RATA-RATA  efreq (seimbang, imb={e.imb.mean():5.1f}) : exact={e.exact.mean():.3f}  w1={e.w1.mean():.3f}  w1_gain={e.w1_gain.mean():+.3f}  macroF1={e.mf1.mean():.3f}")
print(f"  RATA-RATA ewidth (timpang, imb={w.imb.mean():5.1f}) : exact={w.exact.mean():.3f}  w1={w.w1.mean():.3f}  w1_gain={w.w1_gain.mean():+.3f}  macroF1={w.mf1.mean():.3f}")
