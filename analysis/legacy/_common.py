"""Pemilihan folder hasil yang deterministik dan dapat diaudit.

Latar. Setiap DATASET_ID dapat memiliki lebih dari satu folder run, karena
RUN_DIR diberi kunci hash pengaturan: menambah XGBoost/CatBoost/TabPFN membuat
folder baru dan folder lama tetap ada. Memilih dengan `next(...)` mengambil
folder mana pun yang lebih dulu dikembalikan sistem file, sehingga satu analisis
bisa mencampur run dengan dan tanpa booster tanpa ketahuan.

Modul ini memaksa pilihan yang eksplisit dan mencetak apa yang dipilih, agar
percampuran semacam itu tidak dapat terjadi diam-diam.
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"


def _settings(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "evaluation_context.json").read_text(encoding="utf-8"))["settings"]
    except Exception:
        return {}


def selesai(dataset_dir: Path) -> list[Path]:
    """Folder run yang benar-benar tuntas (punya FINAL_COMPLETE.json)."""
    if not dataset_dir.is_dir():
        return []
    return [p for p in dataset_dir.iterdir()
            if p.is_dir() and p.name.startswith("tabular-")
            and (p / "FINAL_COMPLETE.json").is_file()]


def pick_run(dataset_dir: Path, boosters: bool | None = True, verbose: bool = False) -> Path | None:
    """Pilih satu folder run.

    boosters=True  -> hanya run dengan XGBoost/CatBoost/TabPFN tersedia
    boosters=False -> hanya run tanpa ketiganya
    boosters=None  -> abaikan syarat itu

    Bila lebih dari satu memenuhi syarat, diambil yang FINAL_COMPLETE-nya terbaru.
    Mengembalikan None bila tidak ada yang cocok, sehingga pemanggil dapat
    melewati dataset itu secara eksplisit alih-alih memakai folder yang keliru.
    """
    kandidat = selesai(dataset_dir)
    if not kandidat:
        return None
    if boosters is not None:
        cocok = []
        for p in kandidat:
            s = _settings(p)
            punya = bool(s.get("xgboost_available")) and bool(s.get("tabpfn_available"))
            if punya == boosters:
                cocok.append(p)
        kandidat = cocok
    if not kandidat:
        return None
    pilihan = max(kandidat, key=lambda p: (p / "FINAL_COMPLETE.json").stat().st_mtime)
    if verbose:
        s = _settings(pilihan)
        print(f"    {dataset_dir.name:28s} -> {pilihan.name[-17:]}  "
              f"xgb={s.get('xgboost_available')} tabpfn={s.get('tabpfn_available')}")
    return pilihan


def dataset_dirs() -> list[Path]:
    """Folder dataset di runs/, mengabaikan folder bantu."""
    return sorted(p for p in RUNS.iterdir()
                  if p.is_dir() and not p.name.startswith("_") and p.name != "figures")
