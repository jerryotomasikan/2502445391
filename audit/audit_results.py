import ast, hashlib, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, cohen_kappa_score, f1_score

BASE=Path(__file__).parent/'Naskah_Revisi'
OUT=BASE/'audit'
ROOT=REPO_ROOT
sys.stdout.reconfigure(encoding='utf-8')
ns={'np':np,'pd':pd,'RARE_LEVEL_MIN':10}
tree=ast.parse((OUT/'cell_04.py').read_text(encoding='utf-8'))
defs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)]
exec(compile(ast.Module(body=defs,type_ignores=[]),'loader_functions','exec'),ns)
selected=json.loads((OUT/'selected_runs.json').read_text())
CLASSIC={'LogisticRegression','SVM_RBF','RandomForest','ExtraTrees','HistGradientBoosting','XGBoost','CatBoost'}
checks=[]; rows=[]; frames={}; allpred={}; paths={}
for parent in sorted((ROOT/'runs_multiseed').glob('*__seed*')):
    cand=[]
    for p in parent.glob('tabular-*'):
        if (p/'FINAL_COMPLETE.json').exists(): cand.append(p)
    if cand:selected[parent.name]=str(max(cand,key=lambda p:(p/'FINAL_COMPLETE.json').stat().st_mtime))

def check(ok,label):
    checks.append({'ok':bool(ok),'check':label})
    if not ok: print('CHECK FAILED',label)

for id,path in selected.items():
    p=Path(path);paths[id]=p
    get=lambda f:json.loads((p/f).read_text(encoding='utf-8'))
    a=get('data_audit.json');c=get('evaluation_context.json');b=get('binary_selection_lock.json');o=get('ordinal_selection_lock.json');f=get('FINAL_COMPLETE.json')
    for name,sha in f['artifact_sha256'].items():check(hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,f'{id} {name} checksum')
    check(hashlib.sha256(Path(a['source_path']).read_bytes()).hexdigest()==a['source_sha256'],f'{id} source checksum')
    df,features,target=ns['read_dataset_frame'](Path(a['source_path']),a['spec'])
    df=df.apply(pd.to_numeric)
    if a['subsample_applied']:df=df.sample(n=a['subsample_applied'],random_state=c['settings']['seed']).sort_index().reset_index(drop=True)
    ordy,biny,meta=ns['build_targets'](df[target],a['spec'])
    df=df.assign(ordinal_level=ordy,binary_target=biny)
    if a['dedup']:df=df.drop_duplicates(features,keep='first')
    df=df.reset_index(drop=True)
    sha=hashlib.sha256(df[features+['ordinal_level','binary_target']].to_csv(index=False,float_format='%.17g',lineterminator='\n').encode()).hexdigest()
    check(sha==a['clean_sha256'],f'{id} reconstructed data checksum')
    splits=np.load(p/'split_indices.npz')
    if not frames: print('SPLIT KEYS',splits.files)
    pr=pd.read_csv(p/'test_predictions_binary.csv');op=pd.read_csv(p/'test_predictions_ordinal.csv')
    testidx=pr.clean_row_index.to_numpy();devidx=splits['development']
    check(np.array_equal(testidx,splits['test']),f'{id} stored split matches predictions')
    check(np.array_equal(df.iloc[testidx].ordinal_level,pr.quality_true),f'{id} target alignment')
    check(set(testidx).isdisjoint(set(devidx)) and len(testidx)+len(devidx)==len(df),f'{id} disjoint split')
    check(np.array_equal(op.clean_row_index,testidx),f'{id} ordinal row alignment')
    tb=pd.read_csv(p/'test_metrics_binary.csv').set_index('model');to=pd.read_csv(p/'test_metrics_ordinal.csv').set_index('model')
    for m,r in tb.iterrows():
        check(np.isclose(roc_auc_score(pr.binary_true,np.clip(pr['p_'+m],1e-7,1-1e-7)),r.roc_auc,atol=2e-7),f'{id} {m} AUC')
        check(np.isclose(accuracy_score(pr.binary_true,pr['pred_'+m]),r.accuracy),f'{id} {m} accuracy')
    for m,r in to.iterrows():
        y=op.quality_true.to_numpy();yp=op['pred_'+m].to_numpy();score=op['score_'+m].to_numpy()
        check(np.isclose(mean_absolute_error(y,score),r.mae,atol=2e-6),f'{id} {m} MAE')
        check(np.isclose(np.mean(abs(y-yp)<=1),r.within_one_accuracy),f'{id} {m} W1')
        check(np.isclose(accuracy_score(y,yp),r.exact_accuracy),f'{id} {m} exact')
    sel=pd.read_csv(p/'binary_selection_table.csv').sort_values(['roc_auc','f1_macro','balanced_accuracy'],ascending=False)
    cm=sel[sel.model.isin(CLASSIC)].iloc[0].model;nm=sel[sel.model.isin({'MLP','FTTransformer'})].iloc[0].model
    primary=b['primary_binary_model'];runner=b['runner_up_binary_model'];om=o['primary_ordinal_model']
    check(sel[sel.model!='DummyPrior'].iloc[0].model==primary,f'{id} primary development selection')
    yd=df.iloc[devidx].ordinal_level.to_numpy();yt=op.quality_true.to_numpy();levels=np.unique(df.ordinal_level)
    const=int(max(levels,key=lambda k:np.mean(abs(yd-k)<=1)))
    counts=pd.Series(yd).value_counts().sort_index();mode=int(counts.idxmax())
    ov=to.loc[om]
    ac=(pr['pred_'+primary]==pr.binary_true).to_numpy();bc=(pr['pred_'+runner]==pr.binary_true).to_numpy()
    wins=int((ac&~bc).sum());loss=int((~ac&bc).sum());pv=binomtest(wins,wins+loss).pvalue if wins+loss else 1.
    h=pd.util.hash_pandas_object(df[features],index=False)
    overlap=int(h.iloc[testidx].isin(h.iloc[devidx]).sum())
    hf=pd.util.hash_pandas_object(df[features].astype('float32'),index=False)
    overlap32=int(hf.iloc[testidx].isin(hf.iloc[devidx]).sum())
    counts_all=np.array(list(a['ordinal_counts_clean'].values()))
    row=dict(dataset=id,n=len(df),n_raw=a['rows_raw'],p=len(features),n_test=len(yt),seed=c['settings']['seed'],dedup=a['dedup'],duplicates=a['duplicates_removed'],conflicts=a['conflicting_feature_groups'],levels=len(levels),imbalance=counts_all.max()/counts_all.min(),primary=primary,runner=runner,auc=tb.loc[primary,'roc_auc'],acc=tb.loc[primary,'accuracy'],classic=cm,classic_auc=tb.loc[cm,'roc_auc'],neural=nm,neural_auc=tb.loc[nm,'roc_auc'],tabpfn_auc=tb.loc['TabPFN','roc_auc'],ordinal_model=om,mae=ov.mae,exact=ov.exact_accuracy,w1=ov.within_one_accuracy,mf1=ov.macro_f1,qwk=ov.qwk,constant=const,constant_w1=np.mean(abs(yt-const)<=1),dev_mode=mode,mode_w1=np.mean(abs(yt-mode)<=1),mcn_b=wins,mcn_c=loss,mcn_p=pv,test_feature_overlap=overlap,test_float32_overlap=overlap32)
    rows.append(row);frames[id]=(df,devidx,testidx);allpred[id]=(pr,op)
dfout=pd.DataFrame(rows);dfout.to_csv(OUT/'audited_results.csv',index=False)
pairrows=[]
for source in ['abalone','california','concrete','energy','gasturbine','miami','superconduct']:
    a=source+'_efreq';b=source+'_ewidth'
    da,ia,ta=frames[a];db,ib,tb=frames[b]
    pairrows.append(dict(source=source,same_binary=np.array_equal(da.binary_target,db.binary_target),same_test=np.array_equal(ta,tb),shared_test=len(set(ta)&set(tb)),n_test=len(ta)))
pd.DataFrame(pairrows).to_csv(OUT/'binning_split_audit.csv',index=False)
(OUT/'verification_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('VERIFICATION',sum(x['ok'] for x in checks),'/',len(checks))
print(dfout[['dataset','auc','ordinal_model','exact','w1','constant','constant_w1','mcn_b','mcn_c','test_feature_overlap']].to_string(index=False))
print(pd.DataFrame(pairrows).to_string(index=False))

# Additional calculations use the saved development order and never refit the main models.
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def bootstrap_auc(y,ps,seed=314159,B=3000):
    rng=np.random.default_rng(seed);neg=np.flatnonzero(y==0);pos=np.flatnonzero(y==1);values=[]
    for start in range(0,B,100):
        ix=np.concatenate([rng.choice(neg,(100,len(neg))),rng.choice(pos,(100,len(pos)))],axis=1)
        aa=[]
        for p in ps:
            ranks=rankdata(p[ix],axis=1)
            aa.append((ranks[:,len(neg):].sum(axis=1)-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
        values.append(np.array(aa).T)
    return np.concatenate(values)

bootrows=[]
for _,row in dfout[~dfout.dataset.str.contains('__seed')].iterrows():
    id=row.dataset;pr,_=allpred[id];y=pr.binary_true.to_numpy()
    names=[row.classic,row.neural,'TabPFN'];ps=[np.clip(pr['p_'+m].to_numpy(),1e-7,1-1e-7) for m in names]
    bs=bootstrap_auc(y,ps)
    r={'dataset':id}
    for j,label in [(1,'scratch'),(2,'tabpfn')]:
        lo,hi=np.quantile(bs[:,j]-bs[:,0],[.025,.975]);r.update({label+'_delta':roc_auc_score(y,ps[j])-roc_auc_score(y,ps[0]),label+'_lo':lo,label+'_hi':hi})
    bootrows.append(r)
pd.DataFrame(bootrows).to_csv(OUT/'category_bootstrap.csv',index=False)

def search_weights(P,y,seed=20261606):
    if P.shape[1]==1:return np.ones(1),.5
    rng=np.random.default_rng(seed);k=P.shape[1]
    candidates=[np.ones(k)/k]+[w for w in rng.dirichlet(np.ones(k),size=800) if max(w)<=.9 and sum(w>=.05)>=2]
    ranked=sorted([(roc_auc_score(y,P@w),w,P@w) for w in candidates],key=lambda t:t[0],reverse=True)[:30]
    thresholds=np.linspace(.2,.8,241);best=None
    for auc,w,prob in ranked:
        pred=prob[:,None]>=thresholds
        tp=np.sum(pred&(y[:,None]==1),axis=0);tn=np.sum(~pred&(y[:,None]==0),axis=0)
        fp=np.sum(pred&(y[:,None]==0),axis=0);fn=np.sum(~pred&(y[:,None]==1),axis=0)
        f1=tp/(2*tp+fp+fn)+tn/(2*tn+fp+fn);ba=.5*(tp/(tp+fn)+tn/(tn+fp))
        from sklearn.metrics import balanced_accuracy_score
        close=np.flatnonzero(f1>=np.max(f1)-1e-14)
        ix=max(close,key=lambda i:(f1_score(y,pred[:,i],average='macro',zero_division=0),balanced_accuracy_score(y,pred[:,i]),-abs(thresholds[i]-.5)))
        # Match the original scalar metrics exactly for final candidate ties.
        t=thresholds[ix];pp=prob>=t
        from sklearn.metrics import balanced_accuracy_score
        key=(roc_auc_score(y,np.clip(prob,1e-7,1-1e-7)),f1_score(y,pp,average='macro'),balanced_accuracy_score(y,pp))
        if best is None or key>best[0]:best=(key,w,t)
    return best[1],best[2]

abl=[]
for id in json.loads((OUT/'selected_runs.json').read_text()):
    p=paths[id];lock=json.loads((p/'binary_selection_lock.json').read_text());spec=lock['model_specs'];e=spec['SoftVoting'];members=e['members']
    if not {'MLP','FTTransformer'}&set(members):continue
    oof={}
    for directory in ['binary_classic','binary_neural']:
        for f in (p/'cache'/directory).glob('*.json'):
            r=json.loads(f.read_text());fam=r.get('family')
            if fam in members and r.get('status')=='ok' and r['config']==spec[fam]['config'] and f.with_suffix('.npz').exists():
                oof[fam]=np.load(f.with_suffix('.npz'))['probability']
    assert set(oof)==set(members),(id,'missing OOF')
    data,di,ti=frames[id];yd=data.iloc[di].binary_target.to_numpy();pr,_=allpred[id]
    # Validate original development ensemble exactly before counterfactual recombination.
    full_dev=np.column_stack([oof[m] for m in members])@np.array(e['weights'])
    assert np.isclose(roc_auc_score(yd,np.clip(full_dev,1e-7,1-1e-7)),lock['development_metrics']['SoftVoting']['roc_auc'],atol=1e-10)
    full_w,full_t=search_weights(np.column_stack([oof[m] for m in members]),yd,lock['split_seed']+700)
    assert np.allclose(full_w,e['weights'],atol=1e-12),(id,'original ensemble weights not reproduced')
    assert np.isclose(full_t,lock['development_metrics']['SoftVoting']['threshold'],atol=1e-12),(id,'original ensemble threshold not reproduced')
    retained=[m for m in members if m not in {'MLP','FTTransformer'}]
    w,t=search_weights(np.column_stack([oof[m] for m in retained]),yd,lock['split_seed']+700)
    prob=np.clip(np.column_stack([pr['p_'+m] for m in retained])@w,1e-7,1-1e-7)
    metrics=pd.read_csv(p/'test_metrics_binary.csv').set_index('model')
    abl.append(dict(dataset=id,retained=';'.join(retained),neural_weight=sum(v for m,v in zip(members,e['weights']) if m in {'MLP','FTTransformer'}),full_auc=metrics.loc['SoftVoting','roc_auc'],ablated_auc=roc_auc_score(pr.binary_true,prob),full_acc=metrics.loc['SoftVoting','accuracy'],ablated_acc=accuracy_score(pr.binary_true,prob>=t)))
pd.DataFrame(abl).to_csv(OUT/'ensemble_ablation.csv',index=False)
single=[]
for id in ['wine_red','wine_white']:
    data,di,ti=frames[id];model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=6000))
    X=data[['alcohol']].to_numpy(dtype=np.float32);y=data.binary_target.to_numpy();model.fit(X[di],y[di])
    single.append(dict(dataset=id,auc_alcohol=roc_auc_score(y[ti],model.predict_proba(X[ti])[:,1])))
pd.DataFrame(single).to_csv(OUT/'alcohol_baseline.csv',index=False)
print('ADDITIONAL ANALYSIS COMPLETE',len(abl),'ablations; bootstrap',len(bootrows),'configurations')
