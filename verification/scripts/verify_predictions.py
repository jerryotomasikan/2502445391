"""Verify released predictions, paired counts, calibration and fixed-model bootstrap intervals.
Run from any directory. No fitted models, original datasets or training required.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
from scipy.stats import rankdata,binomtest
from sklearn.metrics import roc_auc_score,f1_score,cohen_kappa_score
from sklearn.calibration import calibration_curve
P=Path(__file__).resolve().parents[1];checks=[]
def ck(name,value,detail=None):
 checks.append({'check':name,'passed':bool(value),**({'detail':str(detail)} if detail is not None else {})})
def near(a,b,tol=1e-9):return np.allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def read_csv_exact(path):return pd.read_csv(path,float_precision='round_trip')
def read(n):return read_csv_exact(P/'results'/n)
def boot_auc(y,ps,seed=314159,B=3000,interleaved=False):
 rng=np.random.default_rng(seed);neg=np.flatnonzero(y==0);pos=np.flatnonzero(y==1)
 if interleaved:ix=np.array([np.r_[rng.choice(neg,len(neg)),rng.choice(pos,len(pos))] for _ in range(B)])
 else:ix=np.concatenate([np.concatenate([rng.choice(neg,(100,len(neg))),rng.choice(pos,(100,len(pos)))],axis=1) for _ in range(B//100)])
 auc=[]
 for p in ps:
  a=[]
  for start in range(0,B,100):
   ranks=rankdata(p[ix[start:start+100]],axis=1);a.extend((ranks[:,len(neg):].sum(axis=1)-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
  auc.append(a)
 return np.array(auc).T,ix
allpred={};allstats={};summary=read('audited_results.csv').set_index('dataset');account=read('ordinal_error_accounting.csv').set_index('dataset');inference=read('primary_runner_inference.csv').set_index('dataset');ps=[]
for run in sorted((P/'evidence').iterdir()):
 if not run.is_dir():continue
 n=run.name;bp=read_csv_exact(run/'test_predictions_binary.csv');op=read_csv_exact(run/'test_predictions_ordinal.csv');audit=json.loads((run/'data_audit.json').read_text());ctx=json.loads((run/'evaluation_context.json').read_text());bl=json.loads((run/'binary_selection_lock.json').read_text());ol=json.loads((run/'ordinal_selection_lock.json').read_text());split=np.load(run/'split_indices.npz',allow_pickle=False);rows=read_csv_exact(P/'lineage'/f'{n}_rows.csv');target=json.loads((P/'lineage'/f'{n}_targets.json').read_text());y=bp.binary_true.to_numpy();q=op.quality_true.to_numpy();allpred[n]=bp
 ck(n+' split matches rows',np.array_equal(bp.clean_row_index,split['test']) and not(set(split['test'])&set(split['development'])))
 kept=rows[rows.kept==1].sort_values('clean_row_index');test=kept.set_index('clean_row_index').loc[bp.clean_row_index]
 ck(n+' lineage labels',np.array_equal(y,test.binary_target) and np.array_equal(q,test.ordinal_target))
 ck(n+' lock/context identity',bl['evaluation_context_hash']==ctx['evaluation_context_hash']==ol['evaluation_context_hash'])
 for r in read_csv_exact(run/'test_metrics_binary.csv').to_dict('records'):
  m=r['model'];p=np.clip(bp['p_'+m].to_numpy(),1e-7,1-1e-7);pred=bp['pred_'+m].to_numpy()
  ck(n+' '+m+' binary metrics',near(roc_auc_score(y,p),r['roc_auc']) and near(np.mean(pred==y),r['accuracy']) and near(f1_score(y,pred,average='macro'),r['f1_macro']))
  ck(n+' '+m+' threshold predictions',np.array_equal(pred,(p>=r['threshold']).astype(int)))
 for r in read_csv_exact(run/'test_metrics_ordinal.csv').to_dict('records'):
  m=r['model'];pr=op['pred_'+m].to_numpy();sc=op['score_'+m].to_numpy()
  ck(n+' '+m+' ordinal metrics',near(np.abs(sc-q).mean(),r['mae'],tol=5e-8) and near(np.mean(pr==q),r['exact_accuracy']) and near(np.mean(np.abs(pr-q)<=1),r['within_one_accuracy']) and near(f1_score(q,pr,labels=audit['ordinal_levels'],average='macro',zero_division=0),r['macro_f1']) and near(cohen_kappa_score(q,pr,weights='quadratic'),r['qwk']))
 devq=kept.set_index('clean_row_index').loc[split['development']].ordinal_target.to_numpy();levels=sorted(kept.ordinal_target.unique());const=min(levels,key=lambda c:(-np.mean(np.abs(devq-c)<=1),c));base=read_csv_exact(run/'baseline_predictions.csv');ck(n+' development constant',const==target['constant_selected_on_development'] and np.all(base.constant_prediction==const))
 m=ol['primary_ordinal_model'];a=np.abs(op['pred_'+m].to_numpy()-q)<=1;c=np.abs(q-const)<=1
 if n in account.index:
  r=account.loc[n];ck(n+' published ordinal accounting',near([(~c).sum(),(~a).sum(),(a&~c).sum(),(c&~a).sum(),const],[r.baseline_errors,r.model_errors,r.corrected,r.introduced,r.constant]))
 primary=bl['primary_binary_model'];runner=bl['runner_up_binary_model'];a=bp['pred_'+primary].to_numpy()==y;b=bp['pred_'+runner].to_numpy()==y;bc=int((a&~b).sum());cb=int((~a&b).sum());ep=binomtest(bc,bc+cb,.5).pvalue if bc+cb else 1.
 allstats[n]={'auc':roc_auc_score(y,np.clip(bp['p_'+primary],1e-7,1-1e-7)),'b':bc,'c':cb,'exact_p':ep}
 if n in inference.index:
  r=inference.loc[n];ck(n+' published paired inference',near([bc,cb,ep,100*(a.mean()-b.mean())],[r.b,r.c,r.mcnemar_p,r.accuracy_delta_pp]));ps.append((ep,n,r.holm_p))
 if n in summary.index:
  r=summary.loc[n];ck(n+' published AUC and chosen identifiers',primary==r.primary and ol['primary_ordinal_model']==r.ordinal_model and near(allstats[n]['auc'],r.auc))
 # Recompute all primary/runner-up bootstrap intervals in the original sequential sampling order.
 probs=[np.clip(bp['p_'+k].to_numpy(),1e-7,1-1e-7) for k in [primary,runner]];bs,ix=boot_auc(y,probs,seed=ctx['settings']['seed']+1200,interleaved=True)
 acc=(a[ix].mean(axis=1)-b[ix].mean(axis=1));saved=read_csv_exact(run/'paired_bootstrap_primary_vs_runner.csv').set_index('metric')
 ci=np.quantile(bs[:,0]-bs[:,1],[.025,.975]);ck(n+' primary AUC bootstrap CI',near(ci,saved.loc['ROC-AUC',['ci95_low','ci95_high']].to_numpy(dtype=float)))
 ci=np.quantile(acc,[.025,.975]);ck(n+' primary accuracy bootstrap CI',near(ci,saved.loc['Accuracy',['ci95_low','ci95_high']].to_numpy(dtype=float)))
 print(n,'metrics and bootstrap verified',flush=True)
prev=0
for i,(pv,n,want) in enumerate(sorted(ps)):
 prev=max(prev,min(1,pv*(len(ps)-i)));ck(n+' Holm adjustment',near(prev,want))
# Category bootstrap uses a different, batch-wise random stream, as in the original audit.
for r in read('category_bootstrap.csv').to_dict('records'):
 n=r['dataset'];bp=allpred[n];s=summary.loc[n];y=bp.binary_true.to_numpy();probs=[np.clip(bp['p_'+m].to_numpy(),1e-7,1-1e-7) for m in [s.classic,s.neural,'TabPFN']];bs,_=boot_auc(y,probs)
 for j,label in [(1,'scratch'),(2,'tabpfn')]:
  got=np.r_[roc_auc_score(y,probs[j])-roc_auc_score(y,probs[0]),np.quantile(bs[:,j]-bs[:,0],[.025,.975])];ck(n+' '+label+' category bootstrap',near(got,[r[label+'_delta'],r[label+'_lo'],r[label+'_hi']]))
for r in read('ordinal_baseline_uncertainty.csv').to_dict('records'):
 n=r['dataset'];bp=allpred[n];op=read_csv_exact(P/'evidence'/n/'test_predictions_ordinal.csv');y=bp.binary_true.to_numpy();q=op.quality_true.to_numpy();delta=(np.abs(op['pred_'+r['model']]-q)<=1).astype(int).to_numpy()-(np.abs(q-r['constant'])<=1).astype(int);rng=np.random.default_rng(int(r['seed']));cl=[np.flatnonzero(y==i) for i in [0,1]]
 vals=np.zeros(3000)
 for k in cl:vals+=delta[rng.choice(k,(3000,len(k)))].sum(axis=1)
 vals/=len(y); ck(n+' ordinal baseline bootstrap',near(np.quantile(vals,[.025,.975]),[r['ci_lo'],r['ci_hi']]))
for n in ['wine_red','wine_white']:
 bp=allpred[n];model=json.loads((P/'evidence'/n/'binary_selection_lock.json').read_text())['primary_binary_model'];obs,avg=calibration_curve(bp.binary_true,bp['p_'+model],n_bins=8,strategy='quantile');r=read('calibration_bins.csv').query('dataset==@n');ck(n+' calibration bins',near(avg,r.mean_probability) and near(obs,r.observed_fraction))
for r in read('protocol_comparison.csv').to_dict('records'):
 n=r['source'];ck(n+' protocol AUC difference',near(allstats[n+'_nodedup']['auc']-allstats[n]['auc'],r['delta_auc']))
failed=[c for c in checks if not c['passed']];report={'scope':'Independent recalculation from released saved predictions; no model retraining. Lineage source reconstruction is checked separately.','runs':len(allpred),'bootstrap_replicates':3000,'numeric_tolerances':{'MAE':5e-8,'other_metrics_and_intervals':1e-9,'reason':'Some original float32 ordinal scores were exported as decimal CSV with limited precision. Maximum observed absolute MAE discrepancy is approximately 1.15e-8 grade units, below reported digits. CSV parsing uses round_trip to preserve threshold boundary values.'},'validation_versions':{'numpy':np.__version__,'pandas':pd.__version__},'passed':len(checks)-len(failed),'failed':len(failed),'checks':checks}
(P/'documentation/prediction_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2));print('FAILED',failed[:10]);sys.exit(bool(failed))
