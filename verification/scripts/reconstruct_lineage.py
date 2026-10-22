from pathlib import Path
import ast,json,hashlib,sys,argparse
import numpy as np,pandas as pd
parser=argparse.ArgumentParser(description="Reconstruct lineage using original hash-matched working CSV inputs, without fitting models")
parser.add_argument("--dataset-dir",required=True,type=Path)
args=parser.parse_args();P=Path(__file__).resolve().parents[1]
DATA=args.dataset_dir
source=(P/"scripts/original_target_construction.py").read_text()
module=ast.parse(source);defs=[n for n in ast.walk(module) if isinstance(n,ast.FunctionDef)]
ns={"np":np,"pd":pd,"Path":Path,"RARE_LEVEL_MIN":10};exec(compile(ast.Module(body=defs,type_ignores=[]),"original_target_functions","exec"),ns)
checks=[];environments={};index=[]
def ck(name,value):
 checks.append({'check':name,'passed':bool(value)})
 if not value:raise AssertionError(name)
for run in sorted((P/'evidence').iterdir()):
 if not run.is_dir():continue
 name=run.name;dest=P/'evidence'/name
 audit=json.loads((run/'data_audit.json').read_text());ctx=json.loads((run/'evaluation_context.json').read_text());environments[name]=ctx['package_versions'];spec=audit['spec'];seed=ctx['settings']['seed'];sourcefile=DATA/spec['file']
 ck(name+' source SHA256',hashlib.sha256(sourcefile.read_bytes()).hexdigest()==audit['source_sha256'])
 frame,features,target=ns['read_dataset_frame'](sourcefile,spec)
 frame=frame.apply(pd.to_numeric,errors='coerce');ck(name+' finite input',np.isfinite(frame.to_numpy()).all())
 if spec.get('subsample') and len(frame)>spec['subsample']:frame=frame.sample(n=spec['subsample'],random_state=seed).sort_index()
 source_rows=frame.index.to_numpy();frame=frame.reset_index(drop=True)
 oq,bq,meta=ns['build_targets'](frame[target].to_numpy(),spec);raw=frame.assign(ordinal_level=oq,binary_target=bq)
 groups=raw.groupby(features,sort=False,dropna=False).ngroup().to_numpy();duplicates=raw.duplicated(features,keep='first').to_numpy();kept=~duplicates if audit['dedup'] else np.ones(len(raw),dtype=bool)
 clean=raw.loc[kept].reset_index(drop=True)
 digest=hashlib.sha256(clean[features+['ordinal_level','binary_target']].to_csv(index=False,float_format='%.17g',lineterminator='\n').encode()).hexdigest()
 ck(name+' reconstructed clean SHA256',digest==audit['clean_sha256'])
 splits=np.load(run/'split_indices.npz',allow_pickle=False);dev=splits['development'];test=splits['test'];ck(name+' complete disjoint split',len(set(dev)&set(test))==0 and np.array_equal(np.sort(np.r_[dev,test]),np.arange(len(clean))))
 preds=pd.read_csv(run/'test_predictions_binary.csv');ordp=pd.read_csv(run/'test_predictions_ordinal.csv');ck(name+' prediction row order',np.array_equal(test,preds.clean_row_index) and np.array_equal(test,ordp.clean_row_index))
 ck(name+' binary/ordinal labels',np.array_equal(preds.binary_true,clean.binary_target.to_numpy()[test]) and np.array_equal(ordp.quality_true,clean.ordinal_level.to_numpy()[test]))
 cidx=np.full(len(raw),-1);cidx[kept]=np.arange(kept.sum());split=np.full(len(raw),'discarded_duplicate',dtype=object);part=np.full(len(clean),'development',dtype=object);part[test]='test';split[kept]=part
 lineage=pd.DataFrame({'source_row_index':source_rows,'analysis_row_index':np.arange(len(raw)),'duplicate_group_id':groups,'kept':kept.astype(int),'clean_row_index':cidx,'ordinal_target':oq,'binary_target':bq,'split':split})
 folds=np.load(run/'cv_folds.npz',allow_pickle=False)
 for prefix in ['bin','ord']:
  fold=np.full(len(clean),-1);seen=[]
  for i in range(5):
   tr=folds[f'{prefix}_train_{i}'];va=folds[f'{prefix}_valid_{i}'];ck(name+f' {prefix} fold {i} disjoint',not(set(tr)&set(va)) and np.array_equal(np.sort(np.r_[tr,va]),np.arange(len(dev))))
   fold[dev[va]]=i;seen.extend(va)
  ck(name+f' {prefix} fold coverage',np.array_equal(np.sort(seen),np.arange(len(dev))))
  col=np.full(len(raw),-1);col[kept]=fold;lineage[prefix+'_validation_fold']=col
 lineage.to_csv(P/'lineage'/f'{name}_rows.csv',index=False)
 levels=sorted(clean.ordinal_level.unique());ydev=clean.ordinal_level.to_numpy()[dev];constant=min(levels,key=lambda x:(-np.mean(np.abs(ydev-x)<=1),x))
 pd.DataFrame({'clean_row_index':test,'source_row_index':source_rows[kept][test],'quality_true':ordp.quality_true,'constant_prediction':constant,'within_one_correct':(np.abs(ordp.quality_true-constant)<=1).astype(int)}).to_csv(dest/'baseline_predictions.csv',index=False)
 config={'source_filename':spec['file'],'source_row_index':'0-based data-row index excluding CSV header, before subsampling','original_clean_sha256':audit['clean_sha256'],'reconstructed_clean_sha256':digest,'lineage_status':'Reconstructed from hash-matched source and original target functions; clean checksum and all saved holdout labels match.','seed':seed,'spec':spec,'constant_selected_on_development':int(constant),'original_target_meta':audit['target_meta']}
 if spec['ordinal']['kind']=='bin':
  if spec['ordinal']['strategy']=='quantile':codes,edges=pd.qcut(frame[target],spec['ordinal']['bins'],labels=False,duplicates='drop',retbins=True)
  else:codes,edges=pd.cut(frame[target],spec['ordinal']['bins'],labels=False,include_lowest=True,retbins=True)
  config.update(bin_edges=[float(x) for x in edges],interval_convention='right-closed; first interval includes minimum',raw_bin_to_final_grade={str(int(c)):sorted(set(map(int,oq[np.asarray(codes)==c]))) for c in np.unique(codes)})
 else:config['bin_edges']=None
 rule=spec['binary'];config['binary_threshold']=float(np.median(frame[target])) if rule=='median' else float(rule.split(':')[1]);config['binary_operator']='>' if rule=='median' else '>='
 (P/'lineage'/f'{name}_targets.json').write_text(json.dumps(config,indent=2))
 index.append({'run':name,'dataset_id':audit['dataset_id'],'seed':seed,'rows':len(clean),'test_rows':len(test),'constant':int(constant)})
 print(name,'verified',flush=True)
(P/'documentation/original_package_versions.json').write_text(json.dumps(environments,indent=2));(P/'documentation/lineage_validation.json').write_text(json.dumps({'checks':checks,'passed':len(checks),'failed':0},indent=2));(P/'documentation/run_index.json').write_text(json.dumps(index,indent=2))
print('ALL',len(index),'runs;',len(checks),'lineage checks')
