"""Post hoc paired uncertainty for two illustrative tolerance-baseline contrasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
A=Path(__file__).parent/'results'/'audit'
D=pd.read_csv(A/'audited_results.csv').set_index('dataset')
P=json.loads((A/'selected_runs.json').read_text(encoding='utf-8'))
out=[]
for name in ['wine_red','miami_ewidth']:
    p=Path(P[name]);op=pd.read_csv(p/'test_predictions_ordinal.csv');bp=pd.read_csv(p/'test_predictions_binary.csv');r=D.loc[name]
    assert np.array_equal(op.clean_row_index,bp.clean_row_index)
    assert np.array_equal(op.quality_true,bp.quality_true)
    y=op.quality_true.to_numpy();pr=op['pred_'+r.ordinal_model].to_numpy();c=int(r.constant)
    model=(abs(y-pr)<=1).astype(int);constant=(abs(y-c)<=1).astype(int);delta=model-constant
    assert np.isclose(model.mean(),r.w1) and np.isclose(constant.mean(),r.constant_w1)
    groups=[np.flatnonzero(bp.binary_true.to_numpy()==v) for v in [0,1]]
    rng=np.random.default_rng(314159);bootstrap=np.zeros(3000)
    for g in groups:bootstrap+=delta[rng.choice(g,(3000,len(g)),replace=True)].sum(axis=1)
    bootstrap/=len(y);lo,hi=np.quantile(bootstrap,[.025,.975])
    assert int(model.sum()-constant.sum())==int((delta==1).sum()-(delta==-1).sum())
    # Analytic constant-baseline example used in Methods.
    for k in range(3,30):
        grades=np.arange(k)
        assert np.isclose(max(np.mean(abs(grades-c0)<=1) for c0 in grades),3/k)
    out.append(dict(dataset=name,n=len(y),model=r.ordinal_model,constant=c,
        model_successes=int(model.sum()),constant_successes=int(constant.sum()),
        only_model=int((delta==1).sum()),only_constant=int((delta==-1).sum()),
        difference=float(delta.mean()),ci_lo=float(lo),ci_hi=float(hi),
        bootstrap_replicates=3000,seed=314159,stratification='binary label',selection='fixed from development'))
pd.DataFrame(out).to_csv(A/'ordinal_baseline_uncertainty.csv',index=False)
print(pd.DataFrame(out)[['dataset','only_model','only_constant','difference','ci_lo','ci_hi']].to_string(index=False))
