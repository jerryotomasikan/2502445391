"""Convert the built Word manuscript to OpenDocument text via LibreOffice.

Run after build_manuscript.py. Keeps Paper_Revised.odt in step with the .docx
so the manuscript can be opened and edited without Microsoft Word.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

OUT = Path(__file__).parent / "Naskah_Revisi"
DOCX = OUT / "Paper_Revised.docx"
ODT = OUT / "Paper_Revised.odt"

CANDIDATES = [
    Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
    Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
]

soffice = next((p for p in CANDIDATES if p.exists()), None)
if soffice is None:
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found is None:
        sys.exit("LibreOffice not found; install it or convert Paper_Revised.docx by hand")
    soffice = Path(found)

if not DOCX.exists():
    sys.exit(f"{DOCX.name} is missing; run build_manuscript.py first")

subprocess.run(
    [str(soffice), "--headless", "--convert-to", "odt", "--outdir", str(OUT), str(DOCX)],
    check=True,
    capture_output=True,
)

if not ODT.exists():
    sys.exit("conversion reported success but produced no .odt")

# The conversion is silent on failure modes such as an empty body, so confirm
# the structural landmarks survived rather than trusting the exit code.
content = zipfile.ZipFile(ODT).read("content.xml").decode("utf-8")
tables = content.count("<table:table ")
title = next(z.strip()[2:] for z in (OUT / "Paper_Revised.md").read_text(encoding="utf-8").splitlines() if z.startswith("# "))

problems = []
if tables != 9:
    problems.append(f"expected 9 tables, found {tables}")
if title not in content:
    problems.append("manuscript title missing from the converted document")
for marker in ("TABLE S5", "Abstract"):
    if marker not in content:
        problems.append(f"{marker!r} missing from the converted document")
if problems:
    sys.exit("ODT conversion looks wrong:\n  " + "\n  ".join(problems))

print(f"ODT written: {ODT.name} ({ODT.stat().st_size:,} bytes, {tables} tables)")
