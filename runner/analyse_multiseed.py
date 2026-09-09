r"""
Analisis multi-seed: apakah efek kebocoran duplikat bertahan lintas pembagian data?

Membaca runs_multiseed/<dataset>__seed<N>/ dan melaporkan, untuk tiap seed, selisih
antara lengan tanpa-dedup dan lengan dedup. Yang menentukan bukan rata-ratanya
melainkan simpangan bakunya: bila efek jauh melampaui variasi antar-seed, klaim
bertahan; bila tidak, ia artefak satu pembagian dan harus ditarik.

Sengaja tidak melakukan uji hipotesis pada lima seed. Lima pengamatan tidak cukup
untuk itu, dan paper ini justru mengkritik uji yang tidak bertenaga. Yang dilaporkan
adalah statistik deskriptif dan rentang teramati.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_multiseed"
PAIR = ("wine_red", "wine_red_nodedup")   # (dedup aktif, dedup mati)


def load_run(dataset: str, seed: int):
    """Ambil metrik holdout dari run yang benar-benar selesai."""
    base = RUNS / f"{dataset}__seed{seed}"
    if not base.is_dir():
        return None
    done = [p for p in base.iterdir()
            if p.is_dir() and (p / "FINAL_COMPLETE.json").is_file()]
    if not done:
        return None
    rd = max(done, key=lambda p: (p / "FINAL_COMPLETE.json").stat().st_mtime)
    stat = json.loads((rd / "statistical_summary.json").read_text(encoding="utf-8"))
    binm = pd.read_csv(rd / "test_metrics_binary.csv").set_index("model")
    ordm = pd.read_csv(rd / "test_metrics_ordinal.csv")
    audit = json.loads((rd / "data_audit.json").read_text(encoding="utf-8"))
    primary = stat["primary"]
    return dict(
        auc=float(binm.loc[primary, "roc_auc"]),
        acc=float(binm.loc[primary, "accuracy"]),
        exact=float(ordm.iloc[0]["exact_accuracy"]),
        mae=float(ordm.iloc[0]["mae"]),
        primary=primary, n=audit["rows_clean"],
    )


def main() -> int:
    if not RUNS.is_dir():
        print(f"Folder {RUNS} belum ada. Jalankan run_multiseed.py lebih dulu.")
        return 2

    seeds = sorted({int(p.name.split("__seed")[1]) for p in RUNS.iterdir()
                    if p.is_dir() and "__seed" in p.name})
    if not seeds:
        print("Belum ada run multi-seed.")
        return 2

    dedup_on, dedup_off = PAIR
    rows = []
    for s in seeds:
        a, b = load_run(dedup_on, s), load_run(dedup_off, s)
        if a is None or b is None:
            print(f"  seed {s}: belum lengkap (dedup={'ok' if a else '-'}, "
                  f"nodedup={'ok' if b else '-'}) — dilewati")
            continue
        rows.append(dict(seed=s, auc_on=a["auc"], auc_off=b["auc"], d_auc=b["auc"] - a["auc"],
                         acc_on=a["acc"], acc_off=b["acc"], d_acc=b["acc"] - a["acc"],
                         exact_on=a["exact"], exact_off=b["exact"],
                         d_exact=b["exact"] - a["exact"],
                         model_on=a["primary"], model_off=b["primary"]))
    if not rows:
        print("Tidak ada pasangan seed yang lengkap.")
        return 2

    df = pd.DataFrame(rows)
    df.to_csv(RUNS / "multiseed_leakage.csv", index=False)

    print("=" * 92)
    print("EFEK KEBOCORAN DUPLIKAT LINTAS SEED  (tanpa dedup  minus  dedup)")
    print("=" * 92)
    print(f"{'seed':>10s} {'AUC dedup':>10s} {'AUC bocor':>10s} {'ΔAUC':>9s} "
          f"{'Δacc(pp)':>9s} {'Δexact(pp)':>11s}")
    for _, r in df.iterrows():
        print(f"{r.seed:>10.0f} {r.auc_on:10.4f} {r.auc_off:10.4f} {r.d_auc:+9.4f} "
              f"{100*r.d_acc:+9.2f} {100*r.d_exact:+11.2f}")

    n = len(df)
    print()
    print(f"  jumlah seed lengkap        : {n}")
    print(f"  ΔAUC   rata-rata ± SD      : {df.d_auc.mean():+.4f} ± {df.d_auc.std(ddof=1):.4f}")
    print(f"  ΔAUC   rentang teramati    : {df.d_auc.min():+.4f} .. {df.d_auc.max():+.4f}")
    print(f"  Δexact rata-rata ± SD (pp) : {100*df.d_exact.mean():+.2f} ± {100*df.d_exact.std(ddof=1):.2f}")
    print(f"  Δexact rentang teramati    : {100*df.d_exact.min():+.2f} .. {100*df.d_exact.max():+.2f}")
    print()
    print("  Sebagai pembanding, variasi skor ANTAR-SEED di dalam satu lengan:")
    print(f"    dedup aktif  : SD AUC = {df.auc_on.std(ddof=1):.4f}")
    print(f"    dedup mati   : SD AUC = {df.auc_off.std(ddof=1):.4f}")

    sd_within = max(df.auc_on.std(ddof=1), df.auc_off.std(ddof=1))
    if sd_within > 0:
        print(f"    rasio efek terhadap derau seed = {abs(df.d_auc.mean()) / sd_within:.1f}x")

    print()
    print("=" * 92)
    print("PEMBACAAN")
    print("=" * 92)
    if df.d_auc.min() > 0 and abs(df.d_auc.mean()) > 3 * sd_within:
        print("  Efek positif pada SETIAP seed dan jauh melampaui derau antar-seed.")
        print("  Klaim inflasi kebocoran bertahan terhadap pilihan pembagian data.")
    elif df.d_auc.min() > 0:
        print("  Efek positif pada setiap seed, tetapi tidak jauh dari derau antar-seed.")
        print("  Laporkan rentangnya, jangan hanya satu angka.")
    else:
        print("  Efek TIDAK konsisten lintas seed. Klaim harus ditarik atau dibatasi;")
        print("  angka dari satu pembagian tidak boleh dilaporkan sebagai temuan utama.")
    print()
    print(f"  tersimpan: {RUNS / 'multiseed_leakage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
