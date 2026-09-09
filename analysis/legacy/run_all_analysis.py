r"""
Jalankan seluruh analisis pasca-training secara berurutan.

Membaca artefak di runs/ dan menghasilkan ulang setiap tabel dan gambar yang
dirujuk paper. Aman dijalankan berkali-kali: setiap skrip menimpa keluarannya
sendiri dan tidak menyentuh hasil training.

Urutan mengikuti ketergantungan: agregasi lebih dulu, karena tahap berikutnya
membaca runs/aggregate.csv.

Contoh
------
    python run_all_analysis.py
    python run_all_analysis.py --only aggregate figures
    python run_all_analysis.py --list
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
NB_DIR = ROOT / "Paper"          # beberapa skrip mengeksekusi sel notebook

# (kunci, berkas, cwd, keterangan)
STEPS = [
    ("aggregate",  "aggregate.py",       ROOT,   "Fase 1 — tabel induk runs/aggregate.csv"),
    ("tables",     "agg2.py",            ROOT,   "Fase 1 — tabel ordinal + uji klaim"),
    ("ceiling",    "agg3.py",            ROOT,   "Fase 1 — cek efek langit-langit"),
    ("paired",     "paired_analysis.py", ROOT,   "kontras binning berpasangan (hasil inferensial)"),
    ("nsize",      "n_analysis.py",      ROOT,   "analisis ukuran sampel (deskriptif)"),
    ("stats",      "phase2_stats.py",    ROOT,   "Fase 2a — MDE, TOST, uji berpasangan H1"),
    ("ablation",   "phase2_ablation.py", NB_DIR, "Fase 2b — ablasi ensemble"),
    ("leakage",    "phase2c_leak.py",    ROOT,   "Fase 2c — inflasi kebocoran duplikat"),
    ("baseline",   "item8.py",           NB_DIR, "baseline satu fitur + kurva kalibrasi"),
    ("figures",    "figs.py",            ROOT,   "gambar A, B, C"),
    ("figleak",    "fig_leak.py",        ROOT,   "gambar D — inflasi kebocoran"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="jalankan hanya langkah bernama ini")
    ap.add_argument("--list", action="store_true", help="tampilkan daftar langkah lalu keluar")
    ap.add_argument("--keep-going", action="store_true",
                    help="lanjut walau ada langkah gagal (default: berhenti)")
    args = ap.parse_args()

    if args.list:
        for key, f, _, desc in STEPS:
            print(f"  {key:10s} {f:22s} {desc}")
        return 0

    steps = STEPS
    if args.only:
        names = {k for k, *_ in STEPS}
        unknown = [x for x in args.only if x not in names]
        if unknown:
            print(f"Langkah tidak dikenal: {unknown}\nTersedia: {sorted(names)}", file=sys.stderr)
            return 2
        steps = [s for s in STEPS if s[0] in set(args.only)]

    if not (ROOT / "runs").is_dir():
        print("Folder runs/ belum ada. Jalankan training lebih dulu.", file=sys.stderr)
        return 2

    print(f"Menjalankan {len(steps)} langkah analisis\n")
    hasil, t0 = [], time.perf_counter()
    for i, (key, fname, cwd, desc) in enumerate(steps, 1):
        script = ANALYSIS / fname
        if not script.is_file():
            print(f"  [{i}/{len(steps)}] {key:10s} LEWAT — {script.name} tidak ada")
            hasil.append((key, "LEWAT"))
            continue
        print(f"  [{i}/{len(steps)}] {key:10s} {desc}", flush=True)
        t = time.perf_counter()
        proc = subprocess.run([sys.executable, str(script)], cwd=str(cwd),
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.perf_counter() - t
        if proc.returncode == 0:
            print(f"                 OK ({dt:.0f} d)")
            hasil.append((key, "OK"))
        else:
            print(f"                 GAGAL ({dt:.0f} d)")
            ekor = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            for ln in ekor:
                print(f"                   {ln[:120]}")
            hasil.append((key, "GAGAL"))
            if not args.keep_going:
                print("\nBerhenti pada kegagalan pertama. Pakai --keep-going untuk melanjutkan.")
                break

    gagal = [k for k, s in hasil if s == "GAGAL"]
    print(f"\nSelesai dalam {(time.perf_counter()-t0)/60:.1f} menit")
    print(f"OK: {sum(1 for _, s in hasil if s == 'OK')} | "
          f"gagal: {len(gagal)} | lewat: {sum(1 for _, s in hasil if s == 'LEWAT')}")
    if gagal:
        print("Langkah gagal:", gagal)
    else:
        print("\nKeluaran:")
        for f in ["aggregate.csv", "phase2_statistics.csv", "phase2_ablation.csv",
                  "phase2c_leakage.csv", "paired_binning_contrast.csv",
                  "single_feature_baseline.csv"]:
            p = ROOT / "runs" / f
            if p.is_file():
                print(f"  runs/{f}")
        figs = sorted((ROOT / "runs" / "figures").glob("*.png"))
        for p in figs:
            print(f"  runs/figures/{p.name}")
    return 1 if gagal else 0


if __name__ == "__main__":
    raise SystemExit(main())
