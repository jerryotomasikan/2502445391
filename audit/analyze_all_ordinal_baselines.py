"""Describe paired tolerance errors for every principal task, without retraining.

Model and constant choices come from the existing development-selection audit.
No significance filter or holdout-based selection is applied. These are post hoc
descriptive counts, not independent confirmations across 19 datasets.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'results'
D = pd.read_csv(A / 'audited_results.csv').set_index('dataset')
P = json.loads((A / 'selected_runs.json').read_text(encoding='utf-8'))
order = ['wine_red','wine_red_efreq','wine_white','wine_white_efreq','airquality',
         'concrete_efreq','concrete_ewidth','energy_efreq','energy_ewidth',
         'abalone_efreq','abalone_ewidth','california_efreq','california_ewidth',
         'miami_efreq','miami_ewidth','gasturbine_efreq','gasturbine_ewidth',
         'superconduct_efreq','superconduct_ewidth']
assert set(order) == {s for s in P if not s.endswith('_nodedup')}
checks, rows = [], []
def check(value, label):
    checks.append({'check': label, 'passed': bool(value)})
    if not value:
        raise AssertionError(label)

for name in order:
    p = Path(P[name]); r = D.loc[name]
    source = p / 'test_predictions_ordinal.csv'
    op = pd.read_csv(source); bp = pd.read_csv(p / 'test_predictions_binary.csv')
    check(op.clean_row_index.is_unique, name + ' unique test indices')
    check(np.array_equal(op.clean_row_index, bp.clean_row_index), name + ' row alignment')
    check(np.array_equal(op.quality_true, bp.quality_true), name + ' target alignment')
    y = op.quality_true.to_numpy(); pred = op['pred_' + r.ordinal_model].to_numpy()
    check(np.isfinite(y).all() and np.isfinite(pred).all(), name + ' finite values')
    check(np.equal(pred, np.rint(pred)).all(), name + ' discrete model predictions')
    model = np.abs(y - pred) <= 1; constant = np.abs(y - int(r.constant)) <= 1
    check(np.isclose(model.mean(), r.w1), name + ' reproduce model tolerance accuracy')
    check(np.isclose(constant.mean(), r.constant_w1), name + ' reproduce constant tolerance accuracy')
    check(len(y) == r.n_test, name + ' reproduce test count')
    corrected = int((model & ~constant).sum()); introduced = int((~model & constant).sum())
    base_errors = int((~constant).sum()); model_errors = int((~model).sum())
    both_correct = int((model & constant).sum()); both_wrong = int((~model & ~constant).sum())
    check(both_correct + both_wrong + corrected + introduced == len(y), name + ' full paired partition')
    check(base_errors == corrected + both_wrong, name + ' baseline error decomposition')
    check(model_errors == introduced + both_wrong, name + ' model error decomposition')
    check(base_errors - model_errors == corrected - introduced, name + ' net error identity')
    check(np.isclose((corrected-introduced)/len(y), r.w1-r.constant_w1), name + ' accuracy difference identity')
    rows.append(dict(dataset=name, n=len(y), model=r.ordinal_model, constant=int(r.constant),
        model_w1=float(model.mean()), constant_w1=float(constant.mean()),
        gain_pp=float(100*(model.mean()-constant.mean())), corrected=corrected,
        introduced=introduced, both_correct=both_correct, both_wrong=both_wrong,
        baseline_errors=base_errors, model_errors=model_errors,
        net_error_reduction_pct=100*(base_errors-model_errors)/base_errors if base_errors else None,
        prediction_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))

T = pd.DataFrame(rows).set_index('dataset'); pairs=[]
for source in ['concrete','energy','abalone','california','miami','gasturbine','superconduct']:
    ef, ew = T.loc[source+'_efreq'], T.loc[source+'_ewidth']
    pairs.append(dict(source=source, ew_minus_ef_model_pp=100*(ew.model_w1-ef.model_w1),
        ew_minus_ef_constant_pp=100*(ew.constant_w1-ef.constant_w1),
        ew_minus_ef_gain_pp=ew.gain_pp-ef.gain_pp))
F=pd.DataFrame(pairs)
summary=dict(tasks=len(T), sources=10, positive_net_gains=int((T.gain_pp>0).sum()),
    zero_baseline_error_tasks=T.index[T.baseline_errors==0].tolist(),
    lower_gain_ew=int((F.ew_minus_ef_gain_pp<0).sum()),
    higher_raw_accuracy_ew=int((F.ew_minus_ef_model_pp>1e-10).sum()),
    tied_raw_accuracy_ew=int((abs(F.ew_minus_ef_model_pp)<=1e-10).sum()),
    lower_raw_accuracy_ew=int((F.ew_minus_ef_model_pp < -1e-10).sum()),
    higher_constant_accuracy_ew=int((F.ew_minus_ef_constant_pp>0).sum()),
    interpretation='Post hoc descriptive counts on fixed holdouts; related task variants are dependent; no new inferential tests.')
check(summary['positive_net_gains']==18, '18 tasks improve over fixed constant')
check(summary['zero_baseline_error_tasks']==['wine_red_efreq'], 'only ceiling case has zero baseline errors')
check(summary['lower_gain_ew']==7, 'all seven EW variants have smaller absolute gain')
check(summary['higher_raw_accuracy_ew']==5 and summary['tied_raw_accuracy_ew']==1 and summary['lower_raw_accuracy_ew']==1, 'five higher one tied one lower EW raw score')
T.reset_index().to_csv(A/'ordinal_error_accounting.csv', index=False)
F.to_csv(A/'ordinal_target_comparison.csv', index=False)
(A/'ordinal_error_accounting_summary.json').write_text(json.dumps(summary,indent=2), encoding='utf-8')
(A/'ordinal_error_accounting_checks.json').write_text(json.dumps(checks,indent=2), encoding='utf-8')
print(json.dumps(summary,indent=2)); print('Checks passed:',len(checks))
