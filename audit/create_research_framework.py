"""Draw the executed research workflow, without implying prospective target fitting."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import fontManager
from PIL import Image
from io import BytesIO

OUT = Path(__file__).parent / 'results' / 'Figures'
OUT.mkdir(parents=True, exist_ok=True)
for filename in ['arial.ttf','arialbd.ttf']:
    fontManager.addfont(str(Path(r'C:\Windows\Fonts')/filename))
plt.rcParams.update({'font.family':'Arial', 'pdf.fonttype':42, 'ps.fonttype':42,
                     'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(6.4, 8.448))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0,100), ylim=(0,132)); ax.axis('off')
ink='#000000'; edge='#000000'; blue='#FFFFFF'; teal='#FFFFFF'; amber='#FFFFFF'
checks=[]

def label(x,y,s,size=8,weight='normal',ha='center',color=ink):
    return ax.text(x,y,s,fontsize=size,fontweight=weight,ha=ha,va='center',color=color,
                   linespacing=1.3)

def box(x,y,w,h,title,body,fill=blue,title_size=8.8,body_size=8):
    # All nodes are white, including the formerly shaded notes.
    patch=FancyBboxPatch((x,y),w,h,boxstyle='square,pad=0.0',
                       linewidth=.7,edgecolor=edge,facecolor='white',zorder=2,
                       linestyle=(0,(4,3)) if title=='Target boundaries' else '-')
    ax.add_patch(patch)
    t=label(x+w/2,y+h-2.35,title,title_size,'bold')
    # Reserve a clear line between title and body.
    b=label(x+w/2,y+(h-4.4)/2,body,body_size)
    t.set_zorder(3);b.set_zorder(3)
    checks.append((patch,t,b,title))

def arrow(points):
    for p,q in zip(points[:-2],points[1:-1]):
        ax.plot([p[0],q[0]],[p[1],q[1]],color=edge,lw=.8,zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                                mutation_scale=8,lw=.8,color=edge,zorder=1,
                                shrinkA=0,shrinkB=0))

# Figure title and description belong to the manuscript caption.

box(2,120,96,10,'1  Data sources and preparation',
    '10 public sources; numerical and missing-value checks; retained predictors\n'
    'Four larger sources subsampled to 4,898 rows before duplicate handling')
arrow([(50,120),(50,116)])

box(2,104,60,12,'2  Define binary and ordinal targets',
    'Native thresholds or median-based binary labels\n'
    'Native grades or equal-frequency / equal-width bins\n'
    'Merge sparse ordinal levels')
box(65,104,33,12,'Target boundaries',
    'Data-derived boundaries / merges\n'
    'use all analyzed target values\n'
    'before deduplication and split',title_size=8.1)

arrow([(32,104),(32,102),(25,102),(25,100)])
arrow([(32,104),(32,102),(75,102),(75,100)])
box(2,90,46,10,'3a  Principal configurations',
    '19 tasks; keep the first row\n'
    'per identical predictor vector')
box(52,90,46,10,'3b  Duplicate-retention comparison',
    '4 additional configurations\n'
    'Keep repeated predictor vectors',title_size=8.5)
arrow([(25,90),(25,88),(50,88),(50,86)])
arrow([(75,90),(75,88),(50,88),(50,86)])

box(2,77,96,9,'4  Split each configuration separately: 80% development / 20% holdout',
    'Stratify by ordinal labels; run each configuration separately without pooling',title_size=8.5)
arrow([(50,77),(50,75),(34,75),(34,73)])
arrow([(50,77),(50,75),(85,75),(85,73)])

box(2,54,65,19,'5  Development only',
    '5-fold out-of-fold predictions\n'
    'Preprocessing fitted within each training fold\n'
    'Classical models | neural networks from scratch | pretrained TabPFN\n'
    'Select models, binary thresholds and ensemble weights\n'
    'Select the constant grade using development labels')
box(72,54,26,19,'Reserved holdout',
    '20% of rows\n\nNo model or\nthreshold selection',title_size=8.8)
arrow([(34.5,54),(34.5,50)])
box(2,41,65,9,'6  Freeze development choices',
    'Refit selected predictors on all development rows',fill=teal)
arrow([(34.5,41),(34.5,37)])
arrow([(85,54),(85,37)])

box(2,26,96,11,'7  Apply fixed predictors and evaluate the holdout',
    'Binary: ROC-AUC and accuracy\n'
    'Ordinal: MAE, exact accuracy, within-one accuracy, macro-F1 and weighted kappa',fill=teal)
arrow([(50,26),(50,24),(17.5,24),(17.5,22)])
arrow([(50,26),(50,22)])
arrow([(50,26),(50,24),(82.5,24),(82.5,22)])

box(2,9,31,13,'RQ1  Duplicate protocols',
    'Overlap and score differences\n'
    'Five seeds for red wine\n'
    'Protocol sensitivity',title_size=8.2)
box(34.5,9,31,13,'RQ2  Ordinal interpretation',
    'Within-one vs constant baseline\n'
    'Paired tolerance-error counts\n'
    'Alternative target definitions',title_size=8.1)
box(67,9,31,13,'RQ3  Model comparisons',
    'Development-selected groups\n'
    'Paired bootstrap intervals\n'
    'Exact McNemar comparisons',title_size=8.2)

label(50,5,'Protocol and binning variants use different holdouts; their contrasts are descriptive.',8)
label(50,2.3,'Report baseline coverage, corrected errors, and the errors remaining after prediction.',8)

# Check text containment against rendered boxes, not just nominal character counts.
fig.canvas.draw();renderer=fig.canvas.get_renderer();audit=[]
for patch,title,body,name in checks:
    p=patch.get_window_extent(renderer)
    for text,kind in [(title,'title'),(body,'body')]:
        b=text.get_window_extent(renderer)
        inside=(b.x0>=p.x0+3 and b.x1<=p.x1-3 and b.y0>=p.y0+2 and b.y1<=p.y1-2)
        audit.append({'box':name,'text':kind,'contained':bool(inside)})
        if not inside:raise RuntimeError(f'Text outside box: {name} ({kind}): {b} vs {p}')
    if body.get_window_extent(renderer).y1>=title.get_window_extent(renderer).y0-1:
        raise RuntimeError(f'Title/body overlap: {name}')
base=OUT/'Figure_1_research_framework'
for suffix in ['png','pdf','svg']:
    fig.savefig(base.with_suffix('.'+suffix),dpi=600,facecolor='white',metadata={'Creator':'Research workflow figure'} if suffix=='pdf' else None)
# Publisher-ready bitmap line art, rendered directly at 1200 dpi.
buffer=BytesIO();fig.savefig(buffer,format='png',dpi=1200,facecolor='white');buffer.seek(0)
with Image.open(buffer) as raster:
    bitmap=raster.convert('L').point(lambda p:255 if p>=128 else 0,mode='1')
    bitmap.save(base.with_suffix('.tif'),compression='group4',dpi=(1200,1200))
(OUT.parent/'audit'/'framework_layout_checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('Monochrome framework exported: 6.4 in, Arial 8-8.8 pt, vector PDF/SVG, PNG 600 dpi, bilevel TIFF 1200 dpi; text-box checks',len(audit))
