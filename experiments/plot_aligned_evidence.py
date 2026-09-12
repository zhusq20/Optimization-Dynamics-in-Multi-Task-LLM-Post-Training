"""Regenerate the manuscript's aligned figures and tables from frozen data only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'experiments/aligned_evidence_20260910'
FIG=ROOT/'figures/aligned_evidence_20260910'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':9.5,
                     'axes.titlesize':10,'legend.fontsize':8.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
                     'svg.fonttype':'none','lines.linewidth':1.3,'savefig.dpi':220})
COL={'M-PG':'#0072B2','M-I64-DR':'#D55E00','M-I64-DT':'#009E73','M-I64-GT':'#CC79A7',
     'S-PG':'#333333','S-I64':'#E69F00'}
MARK={'M-PG':'o','M-I64-DR':'s','M-I64-DT':'^','M-I64-GT':'D','S-PG':'v','S-I64':'P'}
DOM=['Math','Code','IF','GPQA']; TASK=['math','code','if','science']
DC=['#0072B2','#D55E00','#009E73','#CC79A7']
LOSSES=['sampled_pg','topk_intersection','teacher_topk','full_vocab']
LL=['PG','I64','Teacher64','Full']
LC=['#0072B2','#D55E00','#CC79A7','#009E73']
CAP=json.loads((DATA/'capability.json').read_text())
SUM=json.loads((DATA/'raw/results_summary.json').read_text())
ROWS=[json.loads(l) for l in (DATA/'raw/measurements.jsonl').read_text().splitlines()]
GEO=json.loads((DATA/'raw/online_context.json').read_text())['geometry']
GEO_SOURCES=['raw/online_context.json']
if (DATA/'online_geometry_20260912.json').is_file():
    GEO+=json.loads((DATA/'online_geometry_20260912.json').read_text())['geometry']
    GEO_SOURCES.append('online_geometry_20260912.json')
COMPS=json.loads((DATA/'paired_comparisons.json').read_text())
DENSITY_EXPERIMENT=next(r for r in json.loads((DATA/'completed_local_followup_20260912.json').read_text())['experiments']
                        if r['experiment']=='density-mpg100-long-bank1042')
DENSITY_ROWS=DENSITY_EXPERIMENT['optimizer']
plots=[]


def save(fig,name,sources):
    # Equal-aspect heatmaps need a second layout pass before label checks.
    fig.canvas.draw()
    fig.canvas.draw()
    fig.set_layout_engine('none')
    renderer=fig.canvas.get_renderer()
    texts=[]
    for ax in fig.axes:
        texts.extend([ax.title,ax.xaxis.label,ax.yaxis.label])
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            labels=[label for value,label in zip(axis.get_ticklocs(),axis.get_ticklabels())
                    if min(limits)<=value<=max(limits) and label.get_visible()]
            texts.extend(labels)
            boxes=[label.get_window_extent(renderer) for label in labels if label.get_text()]
            assert all(not left.overlaps(right) for i,left in enumerate(boxes) for right in boxes[i+1:]),(name,'overlapping tick labels')
    clipped=[]
    for item in texts+fig.legends:
        if not item.get_visible():continue
        box=item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
            clipped.append(item.get_text() if hasattr(item,'get_text') else 'legend')
    assert not clipped,(name,'text outside canvas',clipped)
    for ext in ['pdf','png','svg']:fig.savefig(FIG/(name+'.'+ext))
    panels=[ax for ax in fig.axes if ax.get_label()!='<colorbar>']
    plots.append({'id':name,'sources':sources,'axes':len(fig.axes),
                  'panels':len(panels),'columns':3,
                  'rows':max(ax.get_subplotspec().rowspan.stop for ax in panels),
                  'panel_bounds':[list(ax.get_position().bounds) for ax in panels],
                  'text_within_canvas':True,'overlapping_tick_labels':False})
    plt.close(fig)


def panel_grid(count, row_height=2.05):
    """Keep a three-column reading order without inventing filler analyses."""
    rows=(count+2)//3
    fig,grid=plt.subplots(rows,3,figsize=(6.7,row_height*rows),
                          layout='constrained',squeeze=False)
    axs=grid.flatten()
    for ax in axs[count:]:fig.delaxes(ax)
    return fig,axs[:count]


def style(ax):
    ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True)


def finish(fig):
    for ax in fig.axes:style(ax)


def capability(models=None, name='capability'):
    models=['S-PG','S-I64','M-PG','M-I64-DR'] if models is None else models
    normalization_view=name=='normalization_capability'
    steps={m:{r['step'] for r in CAP if r['model']==m} for m in models}
    groups=[models] if normalization_view else [['S-PG','S-I64'],['M-PG','M-I64-DR']]
    allowed_steps={m:set.intersection(*(steps[g] for g in group)) for group in groups for m in group}
    measures=DOM+(['mean','worst'] if normalization_view else [])
    titles=['MATH-500','LiveCodeBench','IFBench strict','GPQA avg@4']
    if normalization_view:titles+=['Mean score','Worst domain change']
    fig,axs=panel_grid(len(measures),1.85)
    initial=next(r for r in CAP if r['model']=='Initial')
    def score(row,measure):
        if measure=='mean':return 100*np.mean([row['scores'][d] for d in DOM])
        if measure=='worst':return 100*min(row['scores'][d]-initial['scores'][d] for d in DOM)
        return 100*row['scores'][measure]
    ticks=sorted({0}|set.union(*allowed_steps.values()))
    if max(ticks)>=500:ticks=[step for step in ticks if step!=50]
    for ax,d,title in zip(axs,measures,titles):
        baseline=score(initial,d)
        ax.axhline(baseline,color='#777777',ls=':',lw=1)
        ax.plot(0,baseline,'o',color='#777777',ms=3)
        for m in models:
            c=COL[m]
            rr=sorted([r for r in CAP if r['model']==m and r['step'] in allowed_steps[m]],key=lambda r:r['step'])
            assert {r['step'] for r in rr}==allowed_steps[m] and rr,(m,name)
            ax.plot([0]+[r['step'] for r in rr],[baseline]+[score(r,d) for r in rr],
                    marker=MARK[m],color=c,ms=3.3,ls='--' if m.startswith('S') else '-',label=m)
        ax.set(title=title,xlabel='Optimizer updates',ylabel='Change (pp)' if d=='worst' else 'Score (%)',xticks=ticks)
    fig.legend(*axs[0].get_legend_handles_labels(),ncol=len(models),loc='outside lower center',frameon=False)
    finish(fig);save(fig,name,['capability.json'])
    plots[-1]['models']=models
    plots[-1]['evaluated_steps']={m:sorted(allowed_steps[m]) for m in models}
    plots[-1]['measures']=measures


def normalization():
    fig,axs=panel_grid(4,1.7)
    for ax,d in zip(axs,DOM):
        for i,m in enumerate(['M-I64-DT','M-I64-GT']):
            r=next(r for r in COMPS if r['left']=='M-I64-DR' and r['left_step']==50 and r['right']==m and r['domain']==d)
            ax.errorbar(r['delta_pp'],i,xerr=[[r['delta_pp']-r['lo_pp']],[r['hi_pp']-r['delta_pp']]],
                        fmt=MARK[m],color=COL[m],capsize=2,ms=4)
        ax.axvline(0,color='#999999',lw=.8,ls=':')
        ax.set(yticks=[0,1],yticklabels=['DT − DR','GT − DR'],title=d,xlabel='Difference (pp)',ylim=(-.6,1.6))
    finish(fig);save(fig,'normalization50',['paired_comparisons.json'])


def geometry():
    fig,axs=panel_grid(3,2.0)
    models=['M-PG','M-I64-DR']
    common=set.intersection(*({r['step'] for r in GEO if r['model']==m and r['quantity']==q}
                             for m in models for q in ['delta_fp32','delta_bf16']))
    endpoint=max(common)
    for m,c in COL.items():
        if m not in models:continue
        for ax,q,title in zip(axs[:2],['delta_fp32','delta_bf16'],['(a) FP32 cumulative','(b) BF16 cumulative']):
            rr=sorted([r for r in GEO if r['model']==m and r['quantity']==q and r['step'] in common],key=lambda r:r['step'])
            ax.plot([r['step'] for r in rr],[100*(1-r['metrics']['sparsity_at_0']) for r in rr],
                    color=c,marker=MARK[m],ms=3,label=m)
            ax.set(title=title,xlabel='Optimizer updates',ylabel='Nonzero (%)',xticks=[0]+[step for step in sorted(common) if step>1])
        for i,q in enumerate(['delta_fp32','delta_bf16']):
            r=next(r for r in GEO if r['model']==m and r['quantity']==q and r['step']==endpoint)
            axs[2].plot(i+{'M-PG':-.06,'M-I64-DR':.06}[m],100*r['metrics']['energy90_fraction'],MARK[m],color=c,ms=4)
    axs[2].set(title=f'(c) 90% energy at {endpoint}',xticks=[0,1],xticklabels=['FP32','BF16'],
               ylabel='Coordinates (%)',xlim=(-.4,1.4))
    axs[0].legend(frameon=False,fontsize=7.5,loc='lower right')
    finish(fig);save(fig,'cumulative_geometry',GEO_SOURCES)
    plots[-1]['evaluated_steps']={m:sorted(common) for m in models}
    plots[-1]['endpoint_step']=endpoint


def adam():
    fig,axs=panel_grid(3,2.05)
    for i,(loss,label,c) in enumerate(zip(['zero_gradient']+LOSSES,['Zero']+LL,['#777777']+LC)):
        for ax,q in zip(axs[:2],['master','bf16']):
            ss=[r for r in ROWS if r['kind']=='optimizer' and r['loss']==loss and r['mode']=='history' and r['quantity']==q]
            vals=[100*r['metrics']['fraction_above']['0.0'] for r in ss]
            ax.scatter([i]*len(vals),vals,edgecolors=c,s=17,marker='o',facecolors='none')
            ax.plot(i,np.mean(vals),'_',color=c,ms=9)
        axs[0].set(xticks=range(5),xticklabels=['Zero','PG','I64','T64','Full'])
        axs[1].set(xticks=range(5),xticklabels=['Zero','PG','I64','T64','Full'])
    axs[0].set(title='(a) Local FP32 step',ylabel='Nonzero (%)',ylim=(0,100))
    axs[1].set(title='(b) Local BF16 writeback',ylabel='Nonzero (%)',ylim=(0,.17))
    for loss,label,c in zip(LOSSES[:2],LL[:2],LC[:2]):
        for i,mode in enumerate(['history','reset_m','fresh']):
            ss=[r for r in ROWS if r['kind']=='optimizer' and r['loss']==loss and r['mode']==mode and r['quantity']=='bf16']
            vals=[100*r['metrics']['fraction_above']['0.0'] for r in ss]
            offset=-.08 if loss=='sampled_pg' else .08
            axs[2].scatter([i+offset]*len(vals),vals,c=c,s=18,marker='o' if loss=='sampled_pg' else 's',label=label if i==0 else None)
    axs[2].set(title='(c) Change Adam state',xticks=[0,1,2],xticklabels=['Saved','Reset m','Fresh'],ylabel='BF16 nonzero (%)',ylim=(0,.57))
    axs[2].legend(frameon=False,loc='center left');finish(fig)
    save(fig,'adam_precision',['raw/measurements.jsonl'])


def teacher():
    fig,axs=panel_grid(3,2.45)
    for ax,loss,label in zip(axs[:2],LOSSES[:2],LL[:2]):
        for i,cond in enumerate(['same_teacher_different_input','same_input_different_teacher']):
            ss=[r for r in ROWS if r['kind']=='crossed_pair' and r['loss']==loss and r['condition']==cond]
            for bank,c in enumerate(['#0072B2','#D55E00']):
                rr=[r for r in ss if r['bank']==bank]
                ax.scatter(i+np.linspace(-.15,.15,len(rr)),[r['cosine'] for r in rr],s=9,
                           color=c,alpha=.65,marker='o' if bank==0 else '^',label=f'Bank {bank+1}' if i==0 else None)
        ax.axhline(0,color='#999999',ls=':',lw=.8)
        ax.set(title=f'{label}: teacher / input',xticks=[0,1],xticklabels=['Fix\nteacher','Fix\ninput'],ylabel='Gradient cosine',ylim=(-1.08,1.08),xlim=(-.4,1.4))
    axs[0].legend(frameon=False,loc='lower left',fontsize=8)
    for loss,c in zip(LOSSES[:2],LC[:2]):
        for cond,marker in [('routed','o'),('common','^')]:
            rr=[r for r in ROWS if r['kind']=='teacher_pair' and r['loss']==loss and r['condition']==cond]
            axs[2].scatter([r['support']['0.01']['jaccard'] for r in rr],[r['cosine'] for r in rr],
                           color=c,marker=marker,s=14,alpha=.8,label=('PG' if loss=='sampled_pg' else 'I64')+' / '+cond)
    axs[2].axhline(0,color='#999999',ls=':',lw=.8)
    axs[2].set(title='Overlap and direction',xlabel='Gradient top-1% Jaccard',ylabel='Gradient cosine',ylim=(-1.08,1.08),xlim=(0,.8))
    fig.legend(*axs[2].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False,fontsize=8.5);finish(fig)
    save(fig,'teacher_input',['raw/measurements.jsonl'])


def teacher_overlap():
    fig,axs=panel_grid(2,2.0)
    for ax,loss,label in zip(axs,LOSSES[:2],LL[:2]):
        matrix=np.full((4,4),np.nan)
        for i,left in enumerate(TASK):
            for j,right in enumerate(TASK):
                if i==j:continue
                rr=[r for r in ROWS if r['kind']=='teacher_pair' and r['loss']==loss
                    and r['condition']=='routed' and {r['left'],r['right']}=={left,right}]
                assert len(rr)==2
                matrix[i,j]=np.mean([r['support']['0.01']['jaccard'] for r in rr])
        im=ax.pcolormesh(np.arange(5)-.5,np.arange(5)-.5,np.ma.masked_invalid(matrix),
                         vmin=0,vmax=1,cmap='Blues',edgecolors='white',linewidth=.8)
        ax.set_aspect('equal');ax.set_ylim(3.5,-.5)
        for i in range(4):
            for j in range(4):
                ax.text(j,i,'—' if i==j else f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=7.5)
        ax.set(title=f'{label}: routed gradients',xticks=range(4),yticks=range(4),
               xticklabels=['Math','Code','IF','Sci.'],yticklabels=['Math','Code','IF','Sci.'])
        ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.colorbar(im,ax=axs,shrink=.9,label='Top-1% gradient Jaccard')
    save(fig,'teacher_overlap',['raw/measurements.jsonl'])


def density():
    fig,axs=panel_grid(3,2.35)
    bank_markers=['o','^','s','D']
    for i,(loss,c) in enumerate(zip(LOSSES,LC)):
        for ax,q in zip(axs[:2],['master','bf16']):
            rr=sorted([r for r in DENSITY_ROWS if r['kind']=='optimizer' and r['loss']==loss
                       and r['mode']=='history' and r['quantity']==q],key=lambda r:r['bank'])
            assert len(rr)==4 and [r['bank'] for r in rr]==list(range(4))
            vals=[100*r['metrics']['fraction_above']['0.0'] for r in rr]
            for bank,marker in enumerate(bank_markers):
                ax.scatter(i+(bank-1.5)*.18,vals[bank],color=c,marker=marker,s=20)
            ax.plot(i,np.mean(vals),'_',color=c,ms=9)
        for q in ['master','bf16']:
            rr=sorted([r for r in DENSITY_ROWS if r['kind']=='optimizer' and r['loss']==loss
                and r['mode']=='history' and r['quantity']==q],key=lambda r:r['bank'])
            vals=[100*r['metrics']['top_energy']['0.01'] for r in rr]
            for bank,marker in enumerate(bank_markers):
                axs[2].scatter(i+(bank-1.5)*.18,vals[bank],color='#0072B2' if q=='master' else '#D55E00',marker=marker,s=20)
    for ax in axs:ax.set(xticks=range(4),xticklabels=['PG','I64','T64','Full'],xlim=(-.5,3.5))
    axs[0].set(title='(a) FP32 local step',ylabel='Nonzero (%)',ylim=(0,100))
    axs[1].set(title='(b) BF16 local change',ylabel='Nonzero (%)',ylim=(0,.18))
    axs[2].set(title='(c) Top-1% energy',ylabel='Squared step norm (%)',ylim=(0,106))
    axs[2].legend([Line2D([0],[0],color=c) for c in ['#0072B2','#D55E00']],['FP32','BF16'],frameon=False,loc='center',fontsize=8)
    fig.legend([Line2D([0],[0],marker=marker,color='#555555',ls='') for marker in bank_markers],
               [f'Bank {bank+1}' for bank in range(4)],loc='outside lower center',ncol=4,frameon=False,fontsize=8)
    finish(fig);save(fig,'supervision_density',['completed_local_followup_20260912.json',DENSITY_EXPERIMENT['source']])
    plots[-1]['experiment']=DENSITY_EXPERIMENT['experiment']
    plots[-1]['banks']=list(range(4))


def rollout_records(rows):
    records={}
    for r in rows:
        if 'rollout/step' in r['metrics']:
            records.setdefault(int(r['metrics']['rollout/step']),{}).update(r['metrics'])
    return records


def dynamics():
    fig,axs=panel_grid(3,2.05)
    for ax,m in zip(axs.flat,['M-I64-DR','M-I64-DT','M-I64-GT']):
        rr=[json.loads(l) for l in (DATA/f'raw/{m}_rollout.jsonl').read_text().splitlines()]
        rec=rollout_records(rr)
        for task,d,c in zip(TASK,['Math','Code','IF','Science'],DC):
            xy=[(k,v[f'mopd/task/{task}/token_share']*100) for k,v in sorted(rec.items()) if k<50 and f'mopd/task/{task}/token_share' in v]
            assert [k for k,v in xy]==list(range(50))
            ax.plot(*zip(*xy),color=c,lw=.65,label=d)
        ax.axhline(25,color='#777777',ls=':',lw=.8);ax.set(title=m,xlabel='Rollout index',ylabel='Valid token share (%)',ylim=(0,80))
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
    finish(fig);save(fig,'all_token_shares',[f'raw/{m}_rollout.jsonl' for m in ['M-I64-DR','M-I64-DT','M-I64-GT']])
    plots[-1]['rollout_indices']=list(range(50))


def supplemental():
    # Teacher JS is paired with the same bank, loss and teacher pair; no pooled regression.
    fig,axs=panel_grid(4,1.85)
    pairkeys=sorted({(r['left'],r['right']) for r in ROWS if r['kind']=='teacher_js'})
    palette=plt.get_cmap('tab10')
    for ax,(loss,cond) in zip(axs.flat,[(l,c) for l in LOSSES[:2] for c in ['routed','common']]):
        for r in ROWS:
            if r['kind']!='teacher_pair' or r['loss']!=loss or r['condition']!=cond:continue
            j=next(j for j in ROWS if j['kind']=='teacher_js' and j['bank']==r['bank'] and j['left']==r['left'] and j['right']==r['right'])
            ax.scatter(j['mean'],1-r['support']['0.01']['jaccard'],color=palette(pairkeys.index((r['left'],r['right']))),
                       marker='o' if r['bank']==0 else '^',s=22)
        ax.set(title=('PG' if loss=='sampled_pg' else 'I64')+' / '+cond,xlabel='Teacher JS (nats)',ylabel='1 − Jaccard\n(top-1% gradients)',ylim=(0,1))
    fig.legend([Line2D([0],[0],marker='o',ls='',color=palette(i)) for i in range(6)],
               [a+' / '+b for a,b in pairkeys],loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    finish(fig);save(fig,'teacher_js',['raw/measurements.jsonl'])
    # Scale-aware threshold sensitivity for the local simulated writeback.
    fig,axs=panel_grid(2,2.1)
    thresholds=['0.0','1e-08','1e-07','1e-06','1e-05']
    for ax,q in zip(axs,['master','bf16']):
        for loss,label,c in zip(['zero_gradient']+LOSSES,['Zero']+LL,['#777777']+LC):
            rr=[r for r in ROWS if r['kind']=='optimizer' and r['quantity']==q and r['mode']=='history' and r['loss']==loss]
            vals=np.array([[100*r['metrics']['fraction_above'][t] for t in thresholds] for r in rr])
            ax.plot(range(5),vals.mean(0),marker='o',ms=3,color=c,label=label)
            ax.fill_between(range(5),vals.min(0),vals.max(0),color=c,alpha=.13)
        ax.set(title='FP32 local step' if q=='master' else 'BF16 local writeback',xticks=range(5),
               xticklabels=['0',r'$10^{-8}$',r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$'],
               xlabel='Absolute-change threshold',ylabel='Coordinates (%)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=5,frameon=False,fontsize=8)
    finish(fig);save(fig,'thresholds',['raw/measurements.jsonl'])


def tables():
    def write(name,lines,comment=None):
        spec,header={
            'capability_rows.tex':('lrrrrr',r'Configuration & Updates & Math & Code & IF & GPQA'),
            'normalization_rows.tex':('lrrrrrr',r'Reduction & Math & Code & IF & GPQA & Mean & Worst $\Delta$'),
            'density_capability_rows.tex':('lrrrrrr',r'Configuration & Updates & Math & Code & IF & GPQA & Mean'),
            'teacher_rows.tex':('lrrr',r'Teacher / target domain & Responses & Score (\%) & Truncation (\%)'),
            'paired_rows.tex':('llrl',r'Reduction & Domain & Difference (pp) & 95\% interval'),
            'coverage_rows.tex':('rlrrrr',r'Bank & Domain & Mass in $T$ (\%) & Mass in $I$ (\%) & Entropy & $\|g_{\mathrm{PG}}\|_2$'),
            'normalization_fixed_batch_rows.tex':('rrrrr',r'Updates & Batch & DR--DT & DR--GT & DT--GT'),
            'endpoint_paired_rows.tex':('lrlrl',r'Setting & Update & Domain & I64 $-$ PG (pp) & 95\% interval (pp)'),
        }[name]
        (DATA/name.replace('_rows','_table')).write_text('\n'.join(
            ([comment] if comment else [])+[r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule']+
            lines+[r'\bottomrule',r'\end{tabular}'])+'\n')
    order={'Initial':0,'S-PG':1,'S-I64':2,'M-PG':3,'M-I64-DR':4,'M-I64-DT':5,'M-I64-GT':6}
    groups=[['M-I64-DR','M-I64-DT','M-I64-GT'],['S-PG','S-I64'],['M-PG','M-I64-DR']]
    paired={('Initial',0)}
    for models in groups:
        common=set.intersection(*({r['step'] for r in CAP if r['model']==m} for m in models))
        paired|={(m,step) for m in models for step in common}
    rows=sorted([r for r in CAP if (r['model'],r['step']) in paired],key=lambda r:(order[r['model']],r['step']))
    write('capability_rows.tex',[r['model']+' & '+str(r['step'])+' & '+' & '.join(f"{100*r['scores'][d]:.2f}" for d in DOM)+r' \\' for r in rows])
    initial=next(r for r in CAP if r['model']=='Initial')['scores']
    normalization_rows=[]
    for m in ['M-I64-DR','M-I64-DT','M-I64-GT']:
        scores=next(r for r in CAP if r['model']==m and r['step']==50)['scores']
        mean=100*np.mean([scores[d] for d in DOM])
        worst=100*min(scores[d]-initial[d] for d in DOM)
        normalization_rows.append(m.replace('M-I64-','')+' & '+' & '.join(f"{100*scores[d]:.2f}" for d in DOM)
                                  +f' & {mean:.2f} & {worst:+.2f}'+r' \\')
    write('normalization_rows.tex',normalization_rows)
    density_rows=[]
    for models in [['S-PG','S-I64'],['M-PG','M-I64-DR']]:
        steps=set.intersection(*({r['step'] for r in CAP if r['model']==m} for m in models))
        for step in sorted(steps):
            for m in models:
                scores=next(r for r in CAP if r['model']==m and r['step']==step)['scores']
                density_rows.append(m+' & '+str(step)+' & '+' & '.join(f"{100*scores[d]:.2f}" for d in DOM)
                                    +f" & {100*np.mean([scores[d] for d in DOM]):.2f}"+r' \\')
    write('density_capability_rows.tex',density_rows)
    write('teacher_rows.tex',[r['teacher'].replace('teacher_','').capitalize()+' & '+str(r['responses'])+' & '+f"{100*r.get('current_scorer_score',r['score']):.2f}"+' & '+f"{100*r['truncation_rate']:.2f}"+r' \\' for r in json.loads((DATA/'teacher_references.json').read_text())])
    write('paired_rows.tex',[r['right'].replace('M-I64-','')+' & '+r['domain']+' & '+f"{r['delta_pp']:+.2f}"+' & '+f"[{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}]"+r' \\' for r in COMPS if r['left']=='M-I64-DR'])
    coverage=[]
    for r in SUM['coverage']:
        g=next(g for g in ROWS if g['kind']=='gradient_metrics' and g['bank']==r['bank']
               and g['condition']=='routed' and g['loss']=='sampled_pg' and g['teacher']==r['task'])
        # Compact decimal/scientific formatting for the table's large dynamic range.
        def number(x):
            if 1e-3<=x<1e4:return f'{x:.4g}'
            mantissa,exponent=f'{x:.2e}'.split('e')
            return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
        coverage.append(str(r['bank']+1)+' & '+{'math':'Math','code':'Code','if':'IF','science':'Science'}[r['task']]
                        +' & '+f"{100*r['student_mass_in_teacher_top64']:.4f}"
                        +' & '+f"{100*r['student_mass_in_intersection']:.4f}"
                        +' & '+number(r['full_vocabulary_student_entropy'])
                        +' & '+number(g['metrics']['l2'])+r' \\')
    write('coverage_rows.tex',coverage)
    fixed=json.loads((DATA/'normalization_fixed_batch_20260912.json').read_text())['records']
    seeds=sorted({r['bank_seed'] for r in fixed})
    write('normalization_fixed_batch_rows.tex',
          [f"{r['step']} & {seeds.index(r['bank_seed'])+1} & "+' & '.join(
              f"{r['gradient_comparisons'][key]['cosine']:.4f}" for key in ['DR_DT','DR_GT','DT_GT'])+r' \\'
           for r in sorted(fixed,key=lambda r:(r['step'],r['bank_seed']))])
    endpoint_rows=[]
    signed=lambda value:'0.00' if abs(value)<.005 else f'{value:+.2f}'
    for setting,left,right in [('Single','S-PG','S-I64'),('Joint','M-PG','M-I64-DR')]:
        comparisons=[r for r in COMPS if r['left']==left and r['right']==right and r['left_step']==r['right_step']]
        endpoint=max(r['left_step'] for r in comparisons)
        if endpoint_rows:endpoint_rows.append(r'\midrule')
        for domain in DOM:
            row=next(r for r in comparisons if r['left_step']==endpoint and r['domain']==domain)
            endpoint_rows.append(f'{setting} & {endpoint} & {domain} & '+signed(row['delta_pp'])
                                 +f" & [{signed(row['lo_pp'])}, {signed(row['hi_pp'])}]"+r' \\')
    write('endpoint_paired_rows.tex',endpoint_rows,
          '% Source: paired_comparisons.json; complete S500 and M250 I64-minus-PG comparisons.')
    values={'capability':CAP,'paired_comparisons':COMPS,'geometry':GEO,'mechanism':SUM,
            'supervision_density':DENSITY_EXPERIMENT,
            'teacher_pairs':[r for r in ROWS if r['kind']=='teacher_pair'],
            'teacher_js':[r for r in ROWS if r['kind']=='teacher_js']}
    (DATA/'plotted_values.json').write_text(json.dumps(values,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures-only',action='store_true',help='Leave generated LaTeX tables and data exports unchanged.')
    args=parser.parse_args()
    capability();capability(['M-I64-DR','M-I64-DT','M-I64-GT'],'normalization_capability')
    normalization();geometry();adam();teacher();teacher_overlap();density();dynamics();supplemental()
    if not args.figures_only:tables()
    (DATA/'figure_manifest.json').write_text(json.dumps({'figures':plots,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(f'Generated {len(plots)} figures as PDF, PNG and SVG'+('.' if args.figures_only else ', plus eight data tables.'))
