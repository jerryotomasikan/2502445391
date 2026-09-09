"""Butir 8 review: (a) baseline satu fitur, (b) kurva kalibrasi.
Keduanya REFERENSI POST-HOC, dihitung setelah evaluasi terkunci selesai —
dilabeli demikian, sama seperti referensi 'tebak grade 6' pada paper lama."""
import matplotlib; matplotlib.use("Agg")
import sys, io, json, contextlib
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(r"PAPER_ROOT"); RUNS = ROOT / "runs"
NB = json.load(open(ROOT / "Paper" / "Training.ipynb", encoding="utf-8"))
C1, C2, C3 = ("".join(NB["cells"][i]["source"]) for i in (2, 4, 6))

def load_split(ds):
    src = C1.replace("DATASET_ID = 'wine_red'", f"DATASET_ID = {ds!r}")
    g = {"__name__": "__main__", "display": lambda *a, **k: None}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<c1>", "exec"), g); exec(compile(C2, "<c2>", "exec"), g)
        exec(compile(C3, "<c3>", "exec"), g)
    return g

print("=" * 92)
print("(a) BASELINE SATU FITUR — regresi logistik hanya pada 'alcohol' (wine)")
print("=" * 92)
rows = []
for ds in ["wine_red", "wine_white"]:
    g = load_split(ds)
    X, yb, dev, test = g["X_ALL"], g["Y_BINARY"], g["DEV_IDX"], g["TEST_IDX"]
    feats = g["FEATURE_COLUMNS"]; ai = feats.index("alcohol")
    pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=6000))])
    pipe.fit(X[dev][:, [ai]], yb[dev])
    auc1 = roc_auc_score(yb[test], pipe.predict_proba(X[test][:, [ai]])[:, 1])
    # pembanding: seluruh fitur, model utama terkunci
    rd = max((p for p in (RUNS/ds).iterdir() if p.is_dir() and (p/"FINAL_COMPLETE.json").is_file()),
             key=lambda p: (p/"FINAL_COMPLETE.json").stat().st_mtime)
    m = pd.read_csv(rd/"test_metrics_binary.csv").set_index("model")
    st = json.loads((rd/"statistical_summary.json").read_text(encoding="utf-8"))
    full = m.loc[st["primary"], "roc_auc"]
    rows.append(dict(dataset=ds, auc_alcohol=auc1, auc_penuh=full, model=st["primary"],
                     selisih=full-auc1, persen=100*(full-auc1)/(full-0.5)))
    print(f"  {ds:12s} alkohol-saja AUC = {auc1:.4f} | {st['primary']:12s} 11 fitur = {full:.4f} "
          f"| selisih = {full-auc1:+.4f}")
p = pd.DataFrame(rows); p.to_csv(RUNS/"single_feature_baseline.csv", index=False)
print()
print("  Interpretasi: satu fitur sudah menangkap sebagian besar sinyal yang bisa dipakai.")
print(f"  Dari kelebihan AUC di atas tebakan acak (0.5), model 11-fitur hanya menambah "
      f"{100 - p.persen.mean():.0f}% relatif terhadap alkohol saja." if False else
      f"  Sisa yang disumbang 10 fitur lain: {p.selisih.mean():+.4f} AUC rata-rata.")

print()
print("=" * 92)
print("(b) KURVA KALIBRASI — menutup kalimat menggantung soal log loss / Brier")
print("=" * 92)
sets = ["wine_red", "wine_white", "concrete_efreq", "abalone_ewidth"]
fig, axes = plt.subplots(1, len(sets), figsize=(4.0*len(sets), 3.8), dpi=200)
for ax, ds in zip(np.atleast_1d(axes), sets):
    rd = max((q for q in (RUNS/ds).iterdir() if q.is_dir() and (q/"FINAL_COMPLETE.json").is_file()),
             key=lambda q: (q/"FINAL_COMPLETE.json").stat().st_mtime)
    pr = pd.read_csv(rd/"test_predictions_binary.csv")
    st = json.loads((rd/"statistical_summary.json").read_text(encoding="utf-8"))
    met = pd.read_csv(rd/"test_metrics_binary.csv").set_index("model")
    y = pr["binary_true"].to_numpy()
    ax.plot([0, 1], [0, 1], ls=":", c="#888", lw=1)
    for name, col in [(st["primary"], "#4C78A8"), (st["runner_up"], "#E45756")]:
        prob = pr[f"p_{name}"].to_numpy()
        fr, mp = calibration_curve(y, prob, n_bins=8, strategy="quantile")
        ax.plot(mp, fr, "o-", ms=4, lw=1.4, color=col,
                label=f"{name} (Brier {met.loc[name,'brier']:.3f})")
    ax.set(title=ds, xlabel="probabilitas prediksi", ylabel="frekuensi teramati",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=6.5, loc="upper left")
fig.suptitle("Kurva kalibrasi pada locked holdout — model utama vs runner-up", fontsize=11)
fig.tight_layout()
fig.savefig(RUNS/"figures"/"E_calibration.png", bbox_inches="tight")
print("  tersimpan: runs/figures/E_calibration.png  (4 dataset, model utama + runner-up)")
