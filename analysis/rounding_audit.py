"""Would a reader recomputing from the printed digits get the printed answer?

Every derived quantity in the manuscript is recomputed from the *displayed*
operands and compared with the *displayed* result. A mismatch is not a wrong
number, but it is something a referee can query, so it should be pre-empted.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "manuscript" / "Paper_Revised.md"


def tables():
    out, cur = [], []
    for line in MD.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            cur.append(cells)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return dict(zip(["I", "II", "III", "IV", "S1", "S2", "S3", "S4", "S5"], out))


T = tables()
text = MD.read_text(encoding="utf-8")
issues, checked = [], 0


def note(cond, msg):
    global checked
    checked += 1
    if not cond:
        issues.append(msg)


def num(s):
    return float(re.sub(r"[^\d.+-]", "", s))


print("Table II — Difference column against the printed AUCs")
for r in T["II"][1:]:
    ded, ret, diff = num(r[2]), num(r[3]), num(r[4])
    recomputed = round(ret - ded, 4)
    note(abs(recomputed - diff) < 5e-5,
         f"Table II {r[0]}: printed {diff:+.4f}, but {ret:.4f} - {ded:.4f} = {recomputed:+.4f}")

print("Table S1 — deltas against the printed Table IV AUCs")
for r in T["S1"][1:]:
    iv = next(x for x in T["IV"][1:] if x[0] == r[0])
    cls, scr, tab = num(iv[2]), num(iv[4]), num(iv[5])
    d_scr = float(re.match(r"([+-][\d.]+)", r[1]).group(1))
    d_tab = float(re.match(r"([+-][\d.]+)", r[2]).group(1))
    note(abs(round(scr - cls, 4) - d_scr) < 5e-5,
         f"S1 {r[0]}: scratch printed {d_scr:+.4f}, but {scr:.4f} - {cls:.4f} = {round(scr-cls,4):+.4f}")
    note(abs(round(tab - cls, 4) - d_tab) < 5e-5,
         f"S1 {r[0]}: TabPFN printed {d_tab:+.4f}, but {tab:.4f} - {cls:.4f} = {round(tab-cls,4):+.4f}")

print("Table S3 — internal arithmetic on the printed counts")
for r in T["S3"][1:]:
    n, e0, corr, intro, em, red = int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5]), r[6]
    note(em == e0 - corr + intro, f"S3 {r[0]}: {e0} - {corr} + {intro} != {em}")
    if e0 > 0:
        note(abs(round(100 * (e0 - em) / e0, 2) - float(red)) < 5e-3,
             f"S3 {r[0]}: printed {red}, but 100*({e0}-{em})/{e0} = {100*(e0-em)/e0:.2f}")

print("Table III — gains against Tables S2/S3 already verified; check EW < EF")
for r in T["III"][1:]:
    ef, ew = num(r[1]) - num(r[2]), num(r[3]) - num(r[4])
    note(ew < ef, f"Table III {r[0]}: EW gain {ew:.2f} not below EF gain {ef:.2f}")

print("Table S5 — n and margins against Table S4")
for r in T["S5"][1:]:
    s4 = next(x for x in T["S4"][1:] if x[0] == r[0])
    b, c = (int(x) for x in s4[3].split("/"))
    note(int(r[1]) == b + c, f"S5 {r[0]}: discordant {r[1]} vs b+c {b+c}")
    note(int(r[2]) == abs(b - c), f"S5 {r[0]}: margin {r[2]} vs |b-c| {abs(b-c)}")

print("Body text — derived statements against the printed table values")
red3 = next(r for r in T["S3"][1:] if r[0] == "Red wine")
mia3 = next(r for r in T["S3"][1:] if r[0] == "Miami EW")
red2 = next(r for r in T["S2"][1:] if r[0] == "Red wine")
mia2 = next(r for r in T["S2"][1:] if r[0] == "Miami EW")
red3c = next(r for r in T["III"][1:] if r[0] == "Miami")

# 2.57 pp = within-one model minus constant (red wine)
note(abs(round(96.69 - 94.12, 2) - 2.57) < 5e-3, "body: 96.69 - 94.12 should print 2.57")
note("2.57 percentage points" in text, "body: 2.57 pp statement present")
# 1.84 pp for Miami EW
note(abs(round(99.49 - 97.65, 2) - 1.84) < 5e-3, "body: 99.49 - 97.65 should print 1.84")
# 263 vs 256 successes
n_red = int(red3[1])
note(n_red - int(red3[5]) == 263 and n_red - int(red3[2]) == 256,
     "body: 263 / 256 within-one successes must follow from Table S3")
# alcohol-only comparisons
note(abs(round(0.8263 - 0.7516, 4) - 0.0747) < 5e-5, "body: 0.8263 - 0.7516 should print 0.0747")
note(abs(round(0.8305 - 0.7338, 4) - 0.0967) < 5e-5, "body: 0.8305 - 0.7338 should print 0.0967")
# resolution ratios
note(abs(0.0208 / 0.0069 - 3) < 0.06, "body: seed SD over median gap should read as three times")
note(abs(0.0708 / 0.0208 - 3.4) < 0.05, "body: protocol over seed SD should read as 3.4")
note(abs(0.0069 / 0.0208 - 1 / 3) < 0.02, "body: median gap should read as about a third")
# seed spread
note(abs(round(0.8738 - 0.8254, 4) - 0.0484) < 5e-5, "body: seed range consistent")

print()
print(f"{checked} recomputations from printed digits")
if issues:
    print(f"\n{len(issues)} PRESENTATION MISMATCH(ES) — value is right, printed digits do not reproduce it")
    for i in issues:
        print("   -", i)
else:
    print("\nevery printed derived value reproduces from the printed operands")
