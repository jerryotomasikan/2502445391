from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
import json, shutil, re
import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

OUT=Path(__file__).parent/'Naskah_Revisi';A=OUT/'audit';F=OUT/'Figures';F.mkdir(exist_ok=True)
d=pd.read_csv(A/'audited_results.csv').set_index('dataset');main=d[~d.index.str.contains('__seed|nodedup')]
paths=json.loads((A/'selected_runs.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(F/(name+'.'+ext),dpi=600,bbox_inches='tight')
    plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(9.6,3.5),gridspec_kw={'width_ratios':[1.35,1]})
sources=['wine_red','wine_white','concrete_efreq','superconduct_efreq'];names=['Red wine','White wine','Concrete EF','Superconductivity EF'];protocol=[]
for i,(s,label) in enumerate(zip(sources,names)):
    a=d.loc[s];b=d.loc[s+'_nodedup'];axs[0].plot([a.auc,b.auc],[i,i],c='.65',lw=1.5)
    axs[0].scatter(a.auc,i,c='#0072B2',marker='o',s=45,label='Deduplicated' if i==0 else None)
    axs[0].scatter(b.auc,i,c='#D55E00',marker='s',s=45,label='Duplicates retained' if i==0 else None)
    protocol.append(dict(source=s,n_dedup=a.n_test,n_retained=b.n_test,overlap=b.test_feature_overlap,overlap_pct=100*b.test_feature_overlap/b.n_test,auc_dedup=a.auc,auc_retained=b.auc,delta_auc=b.auc-a.auc,exact_dedup=100*a.exact,exact_retained=100*b.exact,delta_exact_pp=100*(b.exact-a.exact)))
axs[0].set(yticks=range(4),yticklabels=names,xlabel='Holdout ROC-AUC',xlim=(.8,1.005),xticks=[.8,.85,.9,.95,1.0]);axs[0].invert_yaxis();axs[0].legend(fontsize=11,loc='lower left');axs[0].set_title('(a) Four protocol comparisons',loc='left',fontsize=13)
seeds=[]
for i,seed in enumerate(range(20260906,20260911)):
    a=d.loc[f'wine_red__seed{seed}'];b=d.loc[f'wine_red_nodedup__seed{seed}'];delta=b.auc-a.auc
    axs[1].plot(delta,i,'o',c='#0072B2');seeds.append(dict(seed=seed,dedup=a.auc,retained=b.auc,delta_auc=delta))
axs[1].set(yticks=range(5),yticklabels=[str(s)[-2:] for s in range(20260906,20260911)],xlabel='ROC-AUC difference',ylabel='Seed suffix',xlim=(0,.10));axs[1].axvline(0,c='.5',lw=.7);axs[1].set_title('(b) Red wine across five seeds',loc='left',fontsize=13)
save(fig,'Figure_1_duplicate_protocols')
pd.DataFrame(protocol).to_csv(A/'protocol_comparison.csv',index=False);pd.DataFrame(seeds).to_csv(A/'seed_comparison.csv',index=False)

sources=['abalone','california','concrete','energy','gasturbine','miami','superconduct'];labels=['Abalone','California','Concrete','Energy','Gas turbine','Miami','Superconductivity'];pairs=[]
fig,axs=plt.subplots(1,2,figsize=(9.6,3.6),sharey=True)
for ax,suffix,title in zip(axs,['_efreq','_ewidth'],['(a) Equal-frequency targets','(b) Equal-width targets']):
    for i,s in enumerate(sources):
        row=d.loc[s+suffix];ax.plot([100*row.constant_w1,100*row.w1],[i,i],c='.7')
        ax.scatter(100*row.constant_w1,i,marker='s',c='#666666',s=32,label='Constant baseline' if i==0 else None)
        ax.scatter(100*row.w1,i,marker='o',c='#0072B2',s=36,label='Selected model' if i==0 else None)
    ax.set(xlim=(40,102),xlabel='Within-one accuracy (%)',yticks=range(7),yticklabels=labels);ax.set_title(title,loc='left',fontsize=13)
axs[0].invert_yaxis();axs[1].legend(fontsize=11,loc='lower left');save(fig,'Figure_2_ordinal_baselines')
for s in sources:
    a=d.loc[s+'_efreq'];b=d.loc[s+'_ewidth'];pairs.append(dict(source=s,ef_levels=a.levels,ew_levels=b.levels,ef_model=100*a.w1,ef_constant=100*a.constant_w1,ef_gain_pp=100*(a.w1-a.constant_w1),ew_model=100*b.w1,ew_constant=100*b.constant_w1,ew_gain_pp=100*(b.w1-b.constant_w1),change_gain_pp=100*((b.w1-b.constant_w1)-(a.w1-a.constant_w1))))
pd.DataFrame(pairs).to_csv(A/'ordinal_contrasts.csv',index=False)

fig,axs=plt.subplots(1,2,figsize=(8.4,3.6))
cal=[]
for ax,s,label in zip(axs,['wine_red','wine_white'],['Red wine','White wine']):
    pr=pd.read_csv(Path(paths[s])/'test_predictions_binary.csv');m=d.loc[s,'primary'];prob=pr['p_'+m].to_numpy();y=pr.binary_true.to_numpy();obs,avg=calibration_curve(y,prob,n_bins=8,strategy='quantile')
    ax.plot([0,1],[0,1],':',c='.5',lw=1);ax.plot(avg,obs,'o-',color='#0072B2',ms=4)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title=label)
    cal.extend([dict(dataset=s,model=m,mean_probability=x,observed_fraction=z) for x,z in zip(avg,obs)])
save(fig,'Figure_3_calibration');pd.DataFrame(cal).to_csv(A/'calibration_bins.csv',index=False)

# Prospective exact McNemar sensitivity scenarios. Discordance probability is
# assumed, not estimated from the realized test outcome. Two-sided alpha=.05.
n=272;ms=np.arange(n+1);lower=[];upper=[]
for m in ms:
    reject=np.array([binomtest(k,m,.5).pvalue<.05 for k in range(m+1)]) if m else np.array([False])
    left=np.flatnonzero(reject & (np.arange(m+1)<m/2));right=np.flatnonzero(reject & (np.arange(m+1)>m/2))
    lower.append(left.max() if len(left) else -1);upper.append(right.min() if len(right) else m+1)
lower=np.array(lower);upper=np.array(upper)
powerrows=[]
for discord in [.05,.10,.20]:
    pm=binom.pmf(ms,n,discord)
    def power(delta):
        q=.5+delta/(2*discord)
        return np.sum(pm*(binom.cdf(lower,ms,q)+binom.sf(upper-1,ms,q)))
    delta=brentq(lambda x:power(x)-.8,0,discord)
    assert .045 < power(0) < .05 or power(0)<.05
    powerrows.append(dict(n=n,discordance=discord,alpha=.05,target_power=.8,min_difference=delta,power=power(delta)))
pd.DataFrame(powerrows).to_csv(A/'prospective_sensitivity.csv',index=False)

# Reuse only PDFs whose actual first-page identities were checked; no renamed
# preprint is presented as the journal publisher's version.
R=OUT/'Reference';R.mkdir(exist_ok=True)
original=REPO_ROOT / r"Paper/Reference"
accepted=['Cao_2020_PRL_CORAL_ordinal.pdf','Gorishniy_2021_NeurIPS_FTTransformer.pdf','Grinsztajn_2022_NeurIPS_trees_outperform.pdf','Kapoor_2023_Patterns_leakage_repro_crisis.pdf','Loshchilov_2019_ICLR_AdamW.pdf','McElfresh_2023_NeurIPS_when_neural_wins.pdf','Shi_2023_PAA_CORN_ordinal.pdf','ShwartzZiv_2022_InfFusion_DL_not_all_need.pdf']
for name in accepted:shutil.copy2(original/name,R/name)
print('Power scenarios',powerrows)
print('Median ordinal change pp',np.median([p['change_gain_pp'] for p in pairs]))
print('Seed mean SD',np.mean([p['delta_auc'] for p in seeds]),np.std([p['delta_auc'] for p in seeds],ddof=1))
