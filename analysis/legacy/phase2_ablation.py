"""Fase 2b: ablasi ensemble — apakah komponen neural benar berkontribusi?
Bangun ulang SoftVoting HANYA dari model klasik, bobot dipilih ULANG pada OOF
development (bukan disalin), lalu terapkan ke holdout. Tidak ada training ulang:
prediksi OOF dan holdout sudah tersimpan."""
import matplotlib; matplotlib.use("Agg")
import sys, io, json, contextlib
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _common import pick_run

# True = pakai run DENGAN XGBoost/CatBoost/TabPFN (hasil akhir paper).
# False = pakai run tanpa ketiganya (arsip pembanding).
# 1/True (default) = run DENGAN XGBoost/CatBoost/TabPFN; 0/False = tanpa ketiganya.
BOOSTERS = __import__("os").environ.get("PAPER_BOOSTERS", "1") not in ("0", "false", "False")
ROOT = REPO_ROOT
RUNS = ROOT / "runs"
NB = json.load(open(ROOT / "Paper" / "Training.ipynb", encoding="utf-8"))
C1, C2, C3 = ("".join(NB["cells"][i]["source"]) for i in (2, 4, 6))
NEURAL = {"MLP", "FTTransformer"}


def dev_labels(dataset):
    """Jalankan Sel 1-3 untuk dataset ini -> Y_DEV (identik dengan run aslinya)."""
    src = C1.replace("DATASET_ID = 'wine_red'", f"DATASET_ID = {dataset!r}")
    g = {"__name__": "__main__", "display": lambda *a, **k: None}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<c1>", "exec"), g)
        exec(compile(C2, "<c2>", "exec"), g)
        exec(compile(C3, "<c3>", "exec"), g)
    return g["Y_DEV"], g["RUN_DIR"]


def choose_threshold(y, p):
    best = None
    for t in np.linspace(0.20, 0.80, 241):
        pr = (p >= t).astype(int)
        key = (f1_score(y, pr, average="macro", zero_division=0),
               balanced_accuracy_score(y, pr), -abs(t - .5))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1]


def search_weights(P, y, seed=20260906 + 700, trials=800):
    """Prosedur bobot IDENTIK Sel 8: bobot sama + 800 draw Dirichlet, saring,
    30 teratas per AUC lalu pilih threshold."""
    k = P.shape[1]
    rng = np.random.default_rng(seed)
    cands = [np.ones(k) / k]
    cands += [w for w in rng.dirichlet(np.ones(k), size=trials)
              if np.max(w) <= .90 and np.sum(w >= .05) >= 2]
    pre = sorted(((roc_auc_score(y, P @ w), i) for i, w in enumerate(cands)), reverse=True)[:30]
    best = None
    for _, i in pre:
        w = cands[i]; p = P @ w
        t = choose_threshold(y, p)
        pr = (p >= t).astype(int)
        key = (roc_auc_score(y, p), f1_score(y, pr, average="macro", zero_division=0),
               balanced_accuracy_score(y, pr))
        if best is None or key > best[0]:
            best = (key, w, t)
    return best[1], best[2]


rows = []
for d in sorted(p for p in RUNS.iterdir() if p.is_dir() and p.name != "figures"):
    rd = pick_run(d, boosters=BOOSTERS)
    if rd is None or not (rd / "binary_selection_lock.json").is_file():
        continue
    lock = json.loads((rd / "binary_selection_lock.json").read_text(encoding="utf-8"))
    specs = lock["model_specs"]
    if "SoftVoting" not in specs:
        continue
    members = specs["SoftVoting"]["members"]
    if not (set(members) & NEURAL):
        rows.append(dict(dataset=d.name, catatan="ensemble asli sudah tanpa neural"))
        continue

    Y_DEV, _ = dev_labels(d.name)
    # OOF tiap anggota, dicocokkan lewat cache json (family + config)
    oof = {}
    for sub in ("binary_classic", "binary_neural"):
        for j in (rd / "cache" / sub).glob("*.json"):
            rec = json.loads(j.read_text(encoding="utf-8"))
            fam = rec.get("family")
            if rec.get("status") != "ok" or fam not in members:
                continue
            if json.dumps(rec["config"], sort_keys=True) != json.dumps(specs[fam]["config"], sort_keys=True):
                continue
            npz = j.with_suffix(".npz")
            if npz.is_file():
                with np.load(npz) as z:
                    oof[fam] = z["probability"].copy()
    klasik = [m for m in members if m not in NEURAL]
    if len(klasik) < 2 or any(m not in oof for m in klasik):
        rows.append(dict(dataset=d.name, catatan=f"OOF tak lengkap ({sorted(oof)})"))
        continue

    w, t = search_weights(np.column_stack([oof[m] for m in klasik]), Y_DEV)

    pred = pd.read_csv(rd / "test_predictions_binary.csv")
    met = pd.read_csv(rd / "test_metrics_binary.csv").set_index("model")
    y = pred["binary_true"].to_numpy()
    p_abl = np.column_stack([pred[f"p_{m}"].to_numpy() for m in klasik]) @ w
    full_auc, full_acc = met.loc["SoftVoting", "roc_auc"], met.loc["SoftVoting", "accuracy"]
    abl_auc = roc_auc_score(y, p_abl)
    abl_acc = accuracy_score(y, p_abl >= t)
    w_neural = sum(wt for m, wt in zip(members, specs["SoftVoting"]["weights"]) if m in NEURAL)

    rows.append(dict(
        dataset=d.name, anggota_neural=[m for m in members if m in NEURAL],
        bobot_neural=round(w_neural, 4), n_klasik=len(klasik),
        auc_penuh=round(full_auc, 4), auc_tanpa_neural=round(abl_auc, 4),
        d_auc=round(full_auc - abl_auc, 4),
        acc_penuh=round(full_acc, 4), acc_tanpa_neural=round(abl_acc, 4),
        d_acc=round(full_acc - abl_acc, 4), catatan="",
    ))
    print(f"  {d.name:22s} bobot neural {w_neural:5.1%} | AUC {full_auc:.4f} -> {abl_auc:.4f} "
          f"(Δ {full_auc-abl_auc:+.4f}) | acc {full_acc:.4f} -> {abl_acc:.4f} (Δ {full_acc-abl_acc:+.4f})",
          flush=True)

df = pd.DataFrame(rows)
df.to_csv(RUNS / "phase2_ablation.csv", index=False)
print(f"\ntersimpan: {RUNS/'phase2_ablation.csv'}")
ok = df[df.catatan == ""]
if len(ok):
    print(f"\n  Dataset diuji            : {len(ok)}")
    print(f"  Ensemble PENUH lebih baik: {(ok.d_auc > 0).sum()}/{len(ok)} (AUC), {(ok.d_acc > 0).sum()}/{len(ok)} (accuracy)")
    print(f"  Median Δ AUC (penuh - tanpa neural) = {ok.d_auc.median():+.4f}")
    print(f"  Median Δ acc                        = {ok.d_acc.median():+.4f}")
    print(f"  Rata-rata bobot neural di ensemble   = {ok.bobot_neural.mean():.1%}")
