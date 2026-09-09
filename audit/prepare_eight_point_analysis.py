"""Verify the manuscript's algebra and compile every recorded null comparison.

This script uses fixed, development-selected predictions. It neither fits models
nor selects examples by their holdout significance.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT=Path(__file__).parent
A = ROOT / 'results'
P=json.loads((A/'selected_runs.json').read_text(encoding='utf-8'))
E=pd.read_csv(A/'ordinal_error_accounting.csv')
D=pd.read_csv(A/'audited_results.csv').set_index('dataset')
checks=[]; rows=[]
def check(value,name):
    checks.append(dict(check=name,passed=bool(value)))
    if not value:raise AssertionError(name)

for _,e in E.iterrows():
    name=e.dataset; p=Path(P[name]); r=D.loc[name]
    b=pd.read_csv(p/'test_predictions_binary.csv'); s=json.loads((p/'statistical_summary.json').read_text(encoding='utf-8'))
    boot=pd.read_csv(p/'paired_bootstrap_primary_vs_runner.csv').set_index('metric')
    y=b.binary_true.to_numpy(); a=b['pred_'+s['primary']].to_numpy()==y; c=b['pred_'+s['runner_up']].to_numpy()==y
    wins=int((a&~c).sum());losses=int((~a&c).sum());n=len(y)
    pv=binomtest(wins,wins+losses).pvalue if wins+losses else 1.
    check(wins==s['mcnemar_b'] and losses==s['mcnemar_c'],name+' recorded discordant counts')
    check(np.isclose(pv,s['mcnemar_exact_p']),name+' exact McNemar p')
    check(np.isclose(a.mean()-c.mean(),boot.loc['Accuracy','delta_primary_minus_runner']),name+' saved accuracy effect')
    check(np.isclose((wins-losses)/n,boot.loc['Accuracy','delta_primary_minus_runner']),name+' paired binary gain identity')
    check(np.isclose(e.model_w1,e.constant_w1+(e.corrected-e.introduced)/e.n),name+' ordinal baseline plus net gain')
    if e.baseline_errors:
        check(np.isclose(e.net_error_reduction_pct/100,(e.model_w1-e.constant_w1)/(1-e.constant_w1)),name+' net error fraction identity')
    else:check(pd.isna(e.net_error_reduction_pct),name+' ceiling fraction undefined')
    check(np.isclose(e.baseline_errors-e.model_errors,e.corrected-e.introduced),name+' counts identity')
    rows.append(dict(dataset=name,n=n,primary=s['primary'],runner=s['runner_up'],
        accuracy_delta_pp=100*boot.loc['Accuracy','delta_primary_minus_runner'],
        accuracy_lo_pp=100*boot.loc['Accuracy','ci95_low'],accuracy_hi_pp=100*boot.loc['Accuracy','ci95_high'],
        auc_delta=boot.loc['ROC-AUC','delta_primary_minus_runner'],auc_lo=boot.loc['ROC-AUC','ci95_low'],auc_hi=boot.loc['ROC-AUC','ci95_high'],
        b=wins,c=losses,discordant=wins+losses,mcnemar_p=pv,
        bootstrap_sha256=hashlib.sha256((p/'paired_bootstrap_primary_vs_runner.csv').read_bytes()).hexdigest()))

N=pd.DataFrame(rows);order=np.argsort(N.mcnemar_p.to_numpy());adjusted=np.minimum(1,np.maximum.accumulate((len(N)-np.arange(len(N)))*N.mcnemar_p.to_numpy()[order]));result=np.empty(len(N));result[order]=adjusted;N['holm_p']=result
check((N.holm_p>=.05).all(),'no primary-runner contrast rejected after Holm')
check(np.isclose(N.holm_p.min(),.22454071044921875),'minimum adjusted p reproduced')
check(sum(E.model_w1>E.constant_w1)==18,'18 of 19 positive ordinal gains')
reduced=E[E.dataset!='airquality']
check(sum(reduced.model_w1>reduced.constant_w1)==17 and len(reduced)==18,'17 of 18 positive gains with air-quality task omitted')
T=pd.read_csv(A/'ordinal_target_comparison.csv')
check((T.ew_minus_ef_gain_pp<0).all(),'seven of seven lower gain contrasts')
check((T.ew_minus_ef_model_pp>1e-10).sum()==5,'five raw-score increases')
check(np.isclose(np.median(-T.ew_minus_ef_gain_pp),25.837320574162675),'median gain decrease reproduced')
# Formula verification at the ceiling and for equally frequent consecutive grades.
for k in range(3,30):
    q=np.arange(k)
    check(np.isclose(max(np.mean(abs(q-v)<=1) for v in q),3/k),f'uniform grade coverage K={k}')
N.to_csv(A/'primary_runner_inference.csv',index=False)
(A/'eight_point_numeric_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('Numeric checks passed',len(checks))
print(N[['dataset','accuracy_delta_pp','accuracy_lo_pp','accuracy_hi_pp','b','c','holm_p']].to_string(index=False))
