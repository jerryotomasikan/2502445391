import sys, io
import pandas as pd, numpy as np
from scipy.stats import spearmanr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
df = pd.read_csv(r"PAPER_ROOT/runs\aggregate.csv")
df["ceiling"] = df[["auc_klasik","auc_neural","auc_sv"]].max(axis=1) >= 0.97
print(f"  rentang n dalam 19 konfigurasi : {df.n.min():,} - {df.n.max():,}  (rasio {df.n.max()/df.n.min():.1f}x)")
for label, s in [("semua 19", df), ("non-ceiling (12)", df[~df.ceiling])]:
    r, p = spearmanr(s.n, s.d_neural)
    print(f"  {label:18s}: Spearman(n, Δ neural-klasik) = {r:+.3f}, p = {p:.3f}")
print()
lo, hi = df[df.n < 2000], df[df.n >= 2000]
print(f"  n < 2000  (k={len(lo)}) : median Δ = {lo.d_neural.median():+.4f}")
print(f"  n >= 2000 (k={len(hi)}) : median Δ = {hi.d_neural.median():+.4f}")
print()
print("  -> Tidak ada kecenderungan neural mengejar seiring n membesar dalam rentang ini.")
print("     Ini BUKAN kurva-n terkendali (dataset berbeda dalam banyak hal), tapi cukup")
print("     untuk menjawab 'apakah temuan ini sekadar efek regime n kecil?' secara deskriptif.")
