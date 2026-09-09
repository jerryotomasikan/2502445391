r"""
Multi-seed tertarget: menguji apakah efek kebocoran duplikat bertahan lintas seed.

Latar. Klaim utama paper adalah besarnya inflasi akibat duplikat yang menyeberang
batas split (+0.078 ROC-AUC pada wine merah). Angka itu berasal dari SATU pembagian
data. Reviewer akan menanyakan apakah ia bertahan bila pembagiannya berbeda.

Rancangan. Jalankan protokol yang sama pada wine_red (dedup aktif) dan
wine_red_nodedup (dedup mati), masing-masing dengan lima seed. Seed mengendalikan
pembagian 80/20, susunan fold, inisialisasi neural, undian bobot ensemble, dan
resampling bootstrap; mengubahnya berarti mengulang seluruh alur, sehingga tidak ada
cache yang terpakai ulang.

Hasilnya bukan sekadar rata-rata: yang dibutuhkan adalah simpangan baku selisih
antar-lengan lintas seed. Bila +0.078 jauh melampaui simpangan itu, klaim bertahan.

Keluaran ditulis ke runs_multiseed/<dataset>__seed<N>/ supaya terpisah dari
runs/ dan tidak mencemari agregasi utama.

Contoh
------
    python run_multiseed.py                 # 2 dataset x 5 seed, 5 proses paralel
    python run_multiseed.py --jobs 4
    python run_multiseed.py --seeds 20260906 20260907
    python run_multiseed.py --force         # ulangi yang sudah selesai
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Paper" / "Training.ipynb"
RUNS = ROOT / "runs_multiseed"

# Pasangan lengan A/B: dedup aktif vs dedup mati, sumber data sama.
DEFAULT_DATASETS = ["wine_red", "wine_red_nodedup"]
# Seed pertama sama dengan run utama, sehingga berfungsi sekaligus sebagai
# pemeriksaan konsistensi terhadap hasil yang sudah ada.
DEFAULT_SEEDS = [20260906, 20260907, 20260908, 20260909, 20260910]

PATCH_DATASET = "DATASET_ID = 'wine_red'"
PATCH_MODE = "MODE = 'research'"
PATCH_SEED = "SEED = 20260906"
PATCH_THREADS = "N_THREADS = max(1, min(18, CPU_COUNT - 2 if CPU_COUNT > 2 else 1))"
PATCH_OUTPUT = "OUTPUT_ROOT = PROJECT_ROOT / 'runs' / DATASET_ID"

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
src = _sub(src, {patch_seed!r}, "SEED = {seed}")
src = _sub(src, {patch_threads!r}, "N_THREADS = {threads}")
src = _sub(src, {patch_output!r},
           "OUTPUT_ROOT = Path(r'{runs}') / (DATASET_ID + '__seed' + str(SEED))")
nb.cells[2].source = src


class _Reporter(ExecutePreprocessor):
    """Lapor mulai/selesai tiap sel ke stderr supaya induk dapat menampilkan progres."""

    def preprocess_cell(self, cell, resources, index):
        is_code = cell.cell_type == "code"
        if is_code:
            print("@@CELL %d START" % index, file=sys.stderr, flush=True)
        out = super().preprocess_cell(cell, resources, index)
        if is_code:
            print("@@CELL %d DONE" % index, file=sys.stderr, flush=True)
        return out


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

SEL_NAMA = {
    2: "konfigurasi", 4: "muat data", 6: "split terkunci", 8: "siapkan klasik",
    10: "cari klasik", 12: "siapkan neural", 14: "cari neural", 16: "kunci+ensemble",
    18: "cari ordinal", 20: "fitting final", 22: "statistik+gambar",
}
AKTIF: "dict[str, float]" = {}
MON_STOP = threading.Event()


def tag(dataset: str, seed: int) -> str:
    return f"{dataset}__seed{seed}"


def _sel_sekarang(log_path: Path) -> str:
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


def _monitor(tiap: int = 60) -> None:
    while not MON_STOP.wait(tiap):
        sedang = sorted(AKTIF.items(), key=lambda kv: kv[1])
        if not sedang:
            continue
        bagian = [f"{n}[{_sel_sekarang(RUNS / n / 'run.log')} {(time.perf_counter() - t) / 60:.0f}m]"
                  for n, t in sedang]
        print(f"    ... {time.strftime('%H:%M:%S')} | berjalan: " + " | ".join(bagian), flush=True)


def run_one(dataset: str, seed: int, mode: str, threads: int, force: bool):
    started = time.perf_counter()
    name = tag(dataset, seed)
    out_dir = RUNS / name
    marker = out_dir / f".done_{mode}"
    if marker.is_file() and not force:
        return name, f"SKIP (sudah selesai, mode {mode})", 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    out_nb = out_dir / "executed.ipynb"
    log_path = out_dir / "run.log"
    runner = out_dir / "_runner.py"
    runner.write_text(CHILD_TEMPLATE.format(
        notebook=NOTEBOOK.as_posix(), nb_parent=NOTEBOOK.parent.as_posix(),
        out_nb=out_nb.as_posix(), runs=RUNS.as_posix(),
        dataset=dataset, mode=mode, seed=seed, threads=threads,
        patch_dataset=PATCH_DATASET, patch_mode=PATCH_MODE, patch_seed=PATCH_SEED,
        patch_threads=PATCH_THREADS, patch_output=PATCH_OUTPUT,
    ), encoding="utf-8")

    AKTIF[name] = time.perf_counter()
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               NUMEXPR_NUM_THREADS=str(threads), PYTHONIOENCODING="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, str(runner)], stdout=log,
                              stderr=subprocess.STDOUT, env=env, cwd=str(NOTEBOOK.parent))
    runner.unlink(missing_ok=True)
    AKTIF.pop(name, None)
    elapsed = time.perf_counter() - started
    if proc.returncode == 0:
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
        return name, "OK", elapsed
    return name, f"GAGAL (exit {proc.returncode}, lihat {log_path.name})", elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--mode", choices=["quick", "research"], default="research")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not NOTEBOOK.is_file():
        print(f"Notebook tidak ditemukan: {NOTEBOOK}", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print("--jobs minimal 1", file=sys.stderr)
        return 2
    if not args.datasets or not args.seeds:
        print("--datasets dan --seeds tidak boleh kosong", file=sys.stderr)
        return 2

    config_src = "".join(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][2]["source"])
    for pat, label in [(PATCH_DATASET, "DATASET_ID"), (PATCH_MODE, "MODE"),
                       (PATCH_SEED, "SEED"), (PATCH_THREADS, "N_THREADS"),
                       (PATCH_OUTPUT, "OUTPUT_ROOT")]:
        if pat not in config_src:
            print(f"Sel 1 tidak memuat pola {label}:\n  {pat}", file=sys.stderr)
            return 2
    registry = re.findall(r"^\s+'([a-z_0-9]+)':\s+dict\(file=", config_src, re.M)
    unknown = [d for d in args.datasets if d not in registry]
    if unknown:
        print(f"DATASET_ID tidak dikenal: {unknown}", file=sys.stderr)
        return 2

    jobs = [(d, s) for s in args.seeds for d in args.datasets]
    threads = args.threads or max(2, max(2, (os.cpu_count() or 4) - 2) // max(1, args.jobs))

    print(f"Notebook   : {NOTEBOOK}")
    print(f"Mode       : {args.mode} | dataset: {args.datasets} | seed: {args.seeds}")
    print(f"Total run  : {len(jobs)} | paralel: {args.jobs} | thread/proses: {threads}")
    print(f"Output     : {RUNS}")
    print("Catatan: seed berbeda berarti split, fold, inisialisasi, dan bobot berbeda,")
    print("         sehingga tidak ada cache yang dipakai ulang. Setiap run dari nol.")
    print(flush=True)

    results, t0 = [], time.perf_counter()
    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(run_one, d, s, args.mode, threads, args.force): (d, s)
                    for d, s in jobs}
            for fut in as_completed(futs):
                name, status, elapsed = fut.result()
                results.append((name, status, elapsed))
                print(f"  [{len(results):>2}/{len(jobs)}] {name:28s} {status:30s} "
                      f"{elapsed/60:6.1f} menit  (sisa {len(jobs)-len(results)})", flush=True)
    finally:
        MON_STOP.set()

    print(f"\nTotal wall-clock: {(time.perf_counter()-t0)/60:.1f} menit")
    failed = [r[0] for r in results if r[1].startswith("GAGAL")]
    print("Gagal:", failed or "tidak ada")
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "run_summary.json").write_text(json.dumps(
        [{"run": n, "status": s, "minutes": round(e / 60, 2)} for n, s, e in sorted(results)],
        indent=2), encoding="utf-8")
    print("Ringkasan  :", RUNS / "run_summary.json")
    print("\nLangkah berikutnya: python analyse_multiseed.py")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
