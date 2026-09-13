"""Clear three-panel alternative for the provisional three-seed normalization data."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(__file__).resolve().parent
FIG=ROOT/'figures/estimated_three_seed_20260913'
summary=list(csv.DictReader((DATA/'estimated_mean_sd.csv').open()))
cap=json.loads((ROOT/'experiments/aligned_evidence_20260910/capability.json').read_text())
models=['M-I64-DR','M-I64-DT','M-I64-GT']
labels={'M-I64-DR':'DR','M-I64-DT':'DT','M-I64-GT':'GT'}
colors={'M-I64-DR':'#D55E00','M-I64-DT':'#009E73','M-I64-GT':'#CC79A7'}
markers={'M-I64-DR':'s','M-I64-DT':'^','M-I64-GT':'D'}
offsets={'M-I64-DR':-.14,'M-I64-DT':0,'M-I64-GT':.14}
domains=['Math','Code','IF','GPQA'];steps=[50,100,250,500]
initial=next(r for r in cap if r['model']=='Initial')['scores']


def record(model,step,measure):
    return next(r for r in summary if r['model']==model and int(r['step'])==step and r['measure']==measure)


plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.titlesize':9.2,
                     'axes.labelsize':8.2,'xtick.labelsize':7.3,'ytick.labelsize':7.3,
                     'legend.fontsize':7.7,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,3,figsize=(6.7,2.35),layout='constrained',
                     gridspec_kw={'width_ratios':[1.05,1.0,1.05]})

# (a) Categorical x positions plus a small horizontal dodge separate overlapping point ranges.
ax=axs[0];positions=np.arange(5);baseline=100*np.mean(list(initial.values()))
ax.axhline(baseline,color='#777777',ls=':',lw=.9)
ax.plot(positions[0],baseline,'o',color='#777777',ms=2.8)
for model in models:
    x=positions[1:]+offsets[model]
    rows=[record(model,step,'mean') for step in steps]
    y=np.array([float(r['mean_pct']) for r in rows]);sd=np.array([float(r['sample_sd_pct']) for r in rows])
    ax.plot(np.r_[positions[0],x],np.r_[baseline,y],color=colors[model],marker=markers[model],ms=3,lw=1.2)
    ax.errorbar(x,y,yerr=sd,fmt='none',ecolor=colors[model],capsize=1.4,elinewidth=.75)
ax.set(title='(a) Mean capability',xlabel='Optimizer updates',ylabel='Score (%)',
       xticks=positions,xticklabels=['0','50','100','250','500'])

# (b) A horizontal endpoint point-range avoids four overlaid trajectory panels.
ax=axs[1];base_y=np.arange(len(domains));
for model in models:
    y=base_y+offsets[model]
    rows=[record(model,500,domain) for domain in domains]
    gain=np.array([float(r['mean_pct'])-100*initial[domain] for r,domain in zip(rows,domains)])
    sd=np.array([float(r['sample_sd_pct']) for r in rows])
    ax.errorbar(gain,y,xerr=sd,fmt=markers[model],color=colors[model],capsize=1.4,ms=3.4,elinewidth=.75)
ax.axvline(0,color='#777777',ls=':',lw=.9)
ax.set(title='(b) Domain gain at update 500',xlabel='Change from initial (pp)',
       yticks=base_y,yticklabels=domains,ylim=(-.55,3.55))
ax.invert_yaxis()

# (c) An annotated heatmap shows when the rules separate without overlapping curves.
ax=axs[2];spread=np.empty((len(domains),len(steps)))
for i,domain in enumerate(domains):
    for j,step in enumerate(steps):
        values=[float(record(model,step,domain)['mean_pct']) for model in models]
        spread[i,j]=max(values)-min(values)
im=ax.imshow(spread,aspect='auto',cmap='Blues',vmin=0,vmax=max(5,spread.max()))
for i in range(len(domains)):
    for j in range(len(steps)):
        color='white' if spread[i,j]>.58*im.norm.vmax else '#222222'
        ax.text(j,i,f'{spread[i,j]:.1f}',ha='center',va='center',fontsize=7,color=color)
ax.set(title='(c) Largest rule gap (pp)',xlabel='Optimizer updates',
       xticks=range(len(steps)),xticklabels=steps,yticks=range(len(domains)),yticklabels=domains)
ax.tick_params(length=0)
for spine in ax.spines.values():spine.set_visible(False)

for ax in axs[:2]:
    ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True)
handles=[Line2D([0],[0],color=colors[m],marker=markers[m],lw=1.2,ms=3,label=labels[m]) for m in models]
fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
for ext in ['pdf','png','svg']:
    fig.savefig(FIG/f'normalization_clear_3seed.{ext}',dpi=260)
plt.close(fig)

caption=(
    'Provisional three-seed visualization. (a) Four-domain mean; small horizontal offsets separate '
    'coincident point ranges. (b) Per-domain change from the shared initial model at update 500. '
    '(c) Largest score gap among DR, DT, and GT at each checkpoint; cell values are percentage points. '
    'Points and bars in (a,b) show mean and one sample standard deviation. PG is excluded because it '
    'changes the supervision objective rather than the averaging rule; its matched DR comparison belongs '
    'in the vocabulary-supervision section. Seeds 43/44 are estimates pending author replacement.')
(DATA/'clear_figure_caption.txt').write_text(caption+'\n')
print(caption)
