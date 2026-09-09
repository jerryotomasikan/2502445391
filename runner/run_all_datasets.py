r"""
Jalankan Training.ipynb untuk banyak DATASET_ID sekaligus, beberapa proses paralel.

Ini pengungkit performa terbesar pada mesin ini: i5-14500 punya 20 logical
processor dan 64 GB RAM, sedangkan satu run dataset kecil tidak pernah memakai
semuanya (RBF-SVM single-thread, model neural mungil). Menjalankan 3-4 dataset
serentak di proses terpisah mengisi mesin jauh lebih baik daripada menaikkan
jumlah thread pada satu run.

Aman karena setiap DATASET_ID menulis ke runs/<DATASET_ID>/ sendiri: cache,
split lock, dan artefak final tidak pernah bertabrakan.

Contoh
------
    # semua dataset, 3 proses paralel (default)
    python run_all_datasets.py

    # cepat: cek alur dulu
    python run_all_datasets.py --mode quick --jobs 4

    # sebagian saja
    python run_all_datasets.py --datasets wine_red wine_white airquality

    # ulangi yang sudah selesai
    python run_all_datasets.py --force
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Paper" / "Training.ipynb"
RUNS = ROOT / "runs"

# Urutan default: yang murah dulu supaya masalah ketahuan cepat.
DEFAULT_DATASETS = [
    "wine_red", "wine_white",
    "wine_red_efreq", "wine_white_efreq",
    "airquality",
    "concrete_efreq", "concrete_ewidth",
    "energy_efreq", "energy_ewidth",
    "abalone_efreq", "abalone_ewidth",
    "miami_efreq", "miami_ewidth",
    "california_efreq", "california_ewidth",
    "gasturbine_efreq", "gasturbine_ewidth",
    "superconduct_efreq", "superconduct_ewidth",
]


PATCH_DATASET = "DATASET_ID = 'wine_red'"
PATCH_MODE = "MODE = 'research'"
PATCH_THREADS = "N_THREADS = max(1, min(18, CPU_COUNT - 2 if CPU_COUNT > 2 else 1))"


CHILD_TEMPLATE = '''import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbformat.read(r"{notebook}", as_version=4)
src = nb.cells[2].source


def _sub(s, old, new):
    if old not in s:
        raise SystemExit("POLA TIDAK DITEMUKAN di Sel 1: " + repr(old))
    return s.replace(old, new, 1)


src = _sub(src, {patch_dataset!r}, "DATASET_ID = {dataset!r}")
src = _sub(src, {patch_mode!r}, "MODE = {mode!r}")
src = _sub(src, {patch_threads!r}, "N_THREADS = {threads}")
nb.cells[2].source = src


class _Reporter(ExecutePreprocessor):
    """Lapor mulai/selesai tiap sel ke stderr supaya induk bisa menampilkan progres."""

    def preprocess_cell(self, cell, resources, index):
        is_code = cell.cell_type == "code"
        if is_code:
            print("@@CELL %d START" % index, file=sys.stderr, flush=True)
        out = super().preprocess_cell(cell, resources, index)
        if is_code:
            print("@@CELL %d DONE" % index, file=sys.stderr, flush=True)
        return out


# allow_errors=False -> berhenti pada error PERTAMA (tanpa kesalahan susulan);
# notebook parsial tetap ditulis lewat finally supaya bisa diperiksa.
ep = _Reporter(timeout=None, kernel_name="python3", allow_errors=False)
failed = None
try:
    ep.preprocess(nb, {{"metadata": {{"path": r"{nb_parent}"}}}})
except Exception as exc:
    failed = "%s: %s" % (type(exc).__name__, exc)
finally:
    nbformat.write(nb, r"{out_nb}")

errs = [(i, o) for i, c in enumerate(nb.cells) if c.cell_type == "code"
        for o in c.get("outputs", []) if o.get("output_type") == "error"]
for i, o in errs:
    print("ERROR cell[%d] %s: %s" % (i, o.get("ename"), o.get("evalue")), file=sys.stderr)
if failed and not errs:
    print("ERROR " + failed, file=sys.stderr)
sys.exit(1 if (errs or failed) else 0)
'''


def build_runner_source(notebook: Path, dataset: str, mode: str, threads: int, out_nb: Path) -> str:
    """Skrip proses anak: patch Sel 1, jalankan notebook, lapor progres per sel.

    Setiap penggantian diverifikasi di sisi anak: kalau pola tidak ditemukan
    (mis. Sel 1 diedit manual), proses berhenti dengan pesan jelas alih-alih
    diam-diam memakai DATASET_ID atau jumlah thread yang salah.
    """
    return CHILD_TEMPLATE.format(
        notebook=notebook, nb_parent=notebook.parent, out_nb=out_nb,
        dataset=dataset, mode=mode, threads=threads,
        patch_dataset=PATCH_DATASET, patch_mode=PATCH_MODE, patch_threads=PATCH_THREADS,
    )


# ---------------------------------------------------------------- progres hidup
# Sel kode notebook ada di indeks genap 2..22 -> "SEL 1/11".."SEL 11/11".
SEL_NAMA = {
    2: "konfigurasi", 4: "muat data", 6: "split terkunci", 8: "siapkan klasik",
    10: "cari klasik", 12: "siapkan neural", 14: "cari neural", 16: "kunci+ensemble",
    18: "cari ordinal", 20: "fitting final", 22: "statistik+gambar",
}
AKTIF: "dict[str, float]" = {}       # dataset -> waktu mulai
MON_STOP = threading.Event()


def _sel_sekarang(log_path: Path) -> str:
    """Baca penanda @@CELL terakhir dari run.log -> 'SEL 5/11 cari klasik'."""
    try:
        ekor = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    except OSError:
        return "memulai"
    idx = None
    for baris in ekor.splitlines():
        if baris.startswith("@@CELL "):
            bagian = baris.split()
            if len(bagian) >= 3 and bagian[1].isdigit():
                idx = int(bagian[1])
    if idx is None:
        return "memulai kernel"
    return f"SEL {idx // 2}/11 {SEL_NAMA.get(idx, '')}".rstrip()


def _monitor(total: int, tiap: int = 60) -> None:
    """Cetak satu baris status berkala supaya terlihat sistem masih bekerja."""
    while not MON_STOP.wait(tiap):
        sedang = sorted(AKTIF.items(), key=lambda kv: kv[1])
        if not sedang:
            continue
        bagian = [
            f"{nama}[{_sel_sekarang(RUNS / nama / 'run.log')} {(time.perf_counter() - t0) / 60:.0f}m]"
            for nama, t0 in sedang
        ]
        print(f"    ... {time.strftime('%H:%M:%S')} | berjalan: " + " | ".join(bagian), flush=True)


def done_marker(dataset: str, mode: str) -> Path:
    """Penanda selesai per (dataset, mode). Run 'quick' tidak boleh membuat run
    'research' dilewati — keduanya memakai RUN_DIR berbeda."""
    return RUNS / dataset / f".done_{mode}"


def run_one(dataset: str, mode: str, threads: int, force: bool) -> tuple[str, str, float]:
    started = time.perf_counter()
    marker = done_marker(dataset, mode)
    if marker.is_file() and not force:
        return dataset, f"SKIP (sudah selesai, mode {mode})", 0.0

    out_dir = RUNS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb = out_dir / "executed.ipynb"
    log_path = out_dir / "run.log"
    # Buang penanda lama dulu: kalau percobaan ini gagal, dataset tidak boleh
    # tetap terhitung "selesai" karena keberhasilan run sebelumnya.
    marker.unlink(missing_ok=True)

    AKTIF[dataset] = time.perf_counter()
    runner = out_dir / "_runner.py"
    runner.write_text(build_runner_source(NOTEBOOK, dataset, mode, threads, out_nb), encoding="utf-8")

    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        NUMEXPR_NUM_THREADS=str(threads),
        PYTHONIOENCODING="utf-8",
    )
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, str(runner)],
            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(NOTEBOOK.parent),
        )
    runner.unlink(missing_ok=True)
    AKTIF.pop(dataset, None)
    elapsed = time.perf_counter() - started
    if proc.returncode == 0:
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
        return dataset, "OK", elapsed
    return dataset, f"GAGAL (exit {proc.returncode}, lihat {log_path.name})", elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, help="DATASET_ID yang dijalankan")
    ap.add_argument("--jobs", type=int, default=3, help="jumlah dataset yang berjalan serentak (default 3)")
    ap.add_argument("--mode", choices=["quick", "research"], default="research")
    ap.add_argument("--threads", type=int, default=0, help="thread per proses; 0 = bagi rata otomatis")
    ap.add_argument("--force", action="store_true", help="jalankan ulang walau sudah ada FINAL_COMPLETE.json")
    args = ap.parse_args()

    if not NOTEBOOK.is_file():
        print(f"Notebook tidak ditemukan: {NOTEBOOK}", file=sys.stderr)
        return 2
    if not args.datasets:                      # '--datasets' tanpa nilai
        args.datasets = list(DEFAULT_DATASETS)
    if args.jobs < 1:
        print("--jobs minimal 1", file=sys.stderr)
        return 2
    seen, ordered = set(), []                  # buang duplikat, jaga urutan
    for d in args.datasets:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    args.datasets = ordered

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    config_src = "".join(nb["cells"][2]["source"])
    for pat, label in [(PATCH_DATASET, "DATASET_ID"), (PATCH_MODE, "MODE"), (PATCH_THREADS, "N_THREADS")]:
        if pat not in config_src:
            print(f"Sel 1 tidak memuat pola {label} yang diharapkan:\n  {pat}\n"
                  f"Kembalikan Sel 1 ke bentuk aslinya atau perbarui skrip ini.", file=sys.stderr)
            return 2
    registry = re.findall(r"^\s+'([a-z_0-9]+)':\s+dict\(file=", config_src, re.M)
    unknown = [d for d in args.datasets if d not in registry]
    if unknown:
        print(f"DATASET_ID tidak dikenal: {unknown}\nTersedia: {sorted(set(registry))}", file=sys.stderr)
        return 2

    total_threads = max(2, (os.cpu_count() or 4) - 2)
    threads = args.threads or max(2, total_threads // max(1, args.jobs))

    print(f"Notebook   : {NOTEBOOK}")
    print(f"Mode       : {args.mode} | dataset: {len(args.datasets)} | paralel: {args.jobs} | thread/proses: {threads}")
    print(f"Output     : {RUNS}\n")

    print("Baris '...' muncul tiap 60 detik selama run berlangsung — itu tanda sistem masih bekerja.", flush=True)
    print(flush=True)

    results, t0 = [], time.perf_counter()
    monitor = threading.Thread(target=_monitor, args=(len(args.datasets),), daemon=True)
    monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_one, d, args.mode, threads, args.force): d for d in args.datasets}
            for fut in as_completed(futures):
                dataset, status, elapsed = fut.result()
                results.append((dataset, status, elapsed))
                sisa = len(args.datasets) - len(results)
                print(f"  [{len(results):>2}/{len(args.datasets)}] {dataset:22s} {status:32s} "
                      f"{elapsed/60:6.1f} menit  (sisa {sisa})", flush=True)
    finally:
        MON_STOP.set()

    print(f"\nTotal wall-clock: {(time.perf_counter() - t0)/60:.1f} menit")
    failed = [r[0] for r in results if r[1].startswith("GAGAL")]
    print("Gagal:", failed or "tidak ada")

    summary = RUNS / "run_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(
        [{"dataset": d, "status": s, "minutes": round(e / 60, 2)} for d, s, e in sorted(results)],
        indent=2), encoding="utf-8")
    print("Ringkasan  :", summary)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
