"""Publication figures from verified summaries only; no fitting or data changes."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).parent; O=ROOT/'Naskah_Revisi'; A=O/'audit'; F=O/'Figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none',
    'lines.linewidth':1,'savefig.facecolor':'white'})
sources={}; checks=[]
def data(name):
    p=A/name;sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p)
def save(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():continue
        box=text.get_window_extent(renderer)
        assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1, (name,text.get_text(),box)
    for ext in ['png','pdf','svg']:
        fig.savefig(F/(name+'.'+ext),dpi=600)
    im=np.asarray(Image.open(F/(name+'.png')).convert('RGB'))
    assert np.array_equal(im[:,:,0],im[:,:,1]) and np.array_equal(im[:,:,1],im[:,:,2])
    checks.append({'figure':name,'grayscale_verified':True,'width_inches':6.4,'png_dpi':600,
        'formats':['png','pdf','svg'],'png_sha256':hashlib.sha256((F/(name+'.png')).read_bytes()).hexdigest()})
    plt.close(fig)

p=data('protocol_comparison.csv'); s=data('seed_comparison.csv')
fig,ax=plt.subplots(1,2,figsize=(6.4,2.65),gridspec_kw={'width_ratios':[1.35,1]})
fig.subplots_adjust(left=.225,right=.96,bottom=.24,top=.85,wspace=.50)
for i,r in enumerate(p.itertuples()):
    ax[0].plot([r.auc_dedup,r.auc_retained],[i,i],color='.6',zorder=1)
    ax[0].plot(r.auc_dedup,i,'o',mfc='white',mec='black',ms=4.5,label='Deduplicated' if i==0 else None)
    ax[0].plot(r.auc_retained,i,'s',mfc='black',mec='black',ms=4,label='Duplicates retained' if i==0 else None)
ax[0].set(yticks=range(4),yticklabels=['Red wine','White wine','Concrete EF','Superconductivity EF'],
    xlabel='Holdout ROC-AUC',xlim=(.80,1.005),xticks=[.8,.9,1.0],ylim=(3.55,-.55))
ax[0].set_title('(a) Protocol comparisons',loc='left',pad=9)
ax[1].plot(s.delta_auc,np.arange(len(s)),'o',color='black',mfc='black',ms=4)
ax[1].set(yticks=range(len(s)),yticklabels=[str(v)[-2:] for v in s.seed],
    xlabel='ROC-AUC difference',ylabel='Seed suffix',xlim=(0,.10),xticks=[0,.05,.10],ylim=(4.55,-.55))
ax[1].set_title('(b) Red-wine seeds',loc='left',pad=9)
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.59,.005))
save(fig,'Figure_2_duplicate_protocols')

p=data('ordinal_contrasts.csv')
labels={'abalone':'Abalone','california':'California','concrete':'Concrete','energy':'Energy',
        'gasturbine':'Gas turbine','miami':'Miami','superconduct':'Superconductivity'}
fig,ax=plt.subplots(1,2,figsize=(6.4,2.85),sharey=True)
fig.subplots_adjust(left=.205,right=.985,bottom=.24,top=.86,wspace=.16)
for a,prefix,title in zip(ax,['ef','ew'],['(a) Equal-frequency targets','(b) Equal-width targets']):
    for i,r in enumerate(p.to_dict('records')):
        a.plot([r[prefix+'_constant'],r[prefix+'_model']],[i,i],color='.6',zorder=1)
        a.plot(r[prefix+'_constant'],i,'s',mfc='white',mec='black',ms=4,label='Constant baseline' if i==0 else None)
        a.plot(r[prefix+'_model'],i,'o',color='black',mfc='black',ms=4,label='Selected model' if i==0 else None)
    a.set(xlim=(40,102),xticks=[40,60,80,100],xlabel='Within-one accuracy (%)',ylim=(6.5,-.5),
        yticks=range(7),yticklabels=[labels[x] for x in p.source])
    a.set_title(title,loc='left',pad=9)
fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.59,.005))
save(fig,'Figure_3_ordinal_baselines')

p=data('calibration_bins.csv')
fig,ax=plt.subplots(1,2,figsize=(6.4,2.95))
fig.subplots_adjust(left=.11,right=.98,bottom=.19,top=.88,wspace=.35)
for a,ds,title in zip(ax,['wine_red','wine_white'],['(a) Red wine','(b) White wine']):
    r=p[p.dataset==ds]
    a.plot([0,1],[0,1],':',color='.45',lw=.8)
    a.plot(r.mean_probability,r.observed_fraction,'o-',color='black',ms=3.5)
    a.set(xlim=(0,1),ylim=(0,1),xticks=[0,.25,.5,.75,1],yticks=[0,.25,.5,.75,1],
        xlabel='Mean predicted probability',ylabel='Observed positive fraction')
    a.set_title(title,loc='left',pad=8)
save(fig,'Figure_4_calibration')
(A/'scientific_figure_checks.json').write_text(json.dumps({'source_hashes':sources,'figures':checks},indent=2),encoding='utf-8')
print('Saved three grayscale figures in PNG 600 dpi, PDF, and SVG; verified source hashes recorded.')
