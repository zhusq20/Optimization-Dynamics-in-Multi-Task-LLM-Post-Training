"""Regenerate the manuscript's aligned figures and tables from frozen data only."""
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
COMPS=json.loads((DATA/'paired_comparisons.json').read_text())
plots=[]


def save(fig,name,sources):
    for ext in ['pdf','png','svg']:fig.savefig(FIG/(name+'.'+ext),bbox_inches='tight')
    plots.append({'id':name,'sources':sources,'axes':len(fig.axes)})
    plt.close(fig)


def style(ax):
    ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True)


def finish(fig):
    for ax in fig.axes:style(ax)


def capability():
    fig,axs=plt.subplots(2,2,figsize=(6.7,3.65),layout='constrained')
    initial=next(r for r in CAP if r['model']=='Initial')
    for ax,d,title in zip(axs.flat,DOM,['MATH-500','LiveCodeBench','IFBench strict','GPQA avg@4']):
        ax.axhline(100*initial['scores'][d],color='#777777',ls=':',lw=1)
        ax.plot(0,100*initial['scores'][d],'o',color='#777777',ms=3)
        for m,c in COL.items():
            rr=sorted([r for r in CAP if r['model']==m],key=lambda r:r['step'])
            ax.plot([0]+[r['step'] for r in rr],[100*initial['scores'][d]]+[100*r['scores'][d] for r in rr],
                    marker=MARK[m],color=c,ms=3.3,ls='--' if m.startswith('S') else '-',label=m)
        ax.set(title=title,xlabel='Optimizer updates',ylabel='Score (%)',xticks=[0,100,250,500])
    fig.legend(*axs[0,0].get_legend_handles_labels(),ncol=6,loc='outside lower center',frameon=False)
    finish(fig);save(fig,'capability',['capability.json'])


def normalization():
    fig,axs=plt.subplots(1,4,figsize=(6.7,1.85),layout='constrained')
    for ax,d in zip(axs,DOM):
        for i,m in enumerate(['M-I64-DT','M-I64-GT']):
            r=next(r for r in COMPS if r['left']=='M-I64-DR' and r['left_step']==50 and r['right']==m and r['domain']==d)
            ax.errorbar(r['delta_pp'],i,xerr=[[r['delta_pp']-r['lo_pp']],[r['hi_pp']-r['delta_pp']]],
                        fmt=MARK[m],color=COL[m],capsize=2,ms=4)
        ax.axvline(0,color='#999999',lw=.8,ls=':')
        ax.set(yticks=[0,1],yticklabels=['DT − DR','GT − DR'],title=d,xlabel='Difference (pp)',ylim=(-.6,1.6))
    finish(fig);save(fig,'normalization50',['paired_comparisons.json'])


def geometry():
    fig,axs=plt.subplots(1,3,figsize=(6.7,2.0),layout='constrained')
    for m,c in COL.items():
        if m.startswith('S'):continue
        for ax,q,title in zip(axs[:2],['delta_fp32','delta_bf16'],['(a) FP32 cumulative','(b) BF16 cumulative']):
            rr=sorted([r for r in GEO if r['model']==m and r['quantity']==q],key=lambda r:r['step'])
            ax.plot([0]+[r['step'] for r in rr],[0]+[100*(1-r['metrics']['sparsity_at_0']) for r in rr],
                    color=c,marker=MARK[m],ms=3,label=m)
            ax.set(title=title,xlabel='Optimizer updates',ylabel='Nonzero (%)',xticks=[0,50,100])
        if m=='M-I64-DT':continue
        for i,q in enumerate(['delta_fp32','delta_bf16']):
            r=next(r for r in GEO if r['model']==m and r['quantity']==q and r['step']==100)
            axs[2].plot(i+{'M-PG':-.06,'M-I64-DR':0,'M-I64-GT':.06}[m],100*r['metrics']['energy90_fraction'],MARK[m],color=c,ms=4)
    axs[2].set(title='(c) 90% energy at 100',xticks=[0,1],xticklabels=['FP32','BF16'],
               ylabel='Coordinates (%)',xlim=(-.4,1.4))
    axs[0].legend(frameon=False,fontsize=7.5,loc='lower right')
    finish(fig);save(fig,'cumulative_geometry',['raw/online_context.json'])


def adam():
    fig,axs=plt.subplots(1,3,figsize=(6.7,2.05),layout='constrained')
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
    fig,axs=plt.subplots(1,3,figsize=(6.7,2.45),layout='constrained')
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


def density():
    fig,axs=plt.subplots(1,3,figsize=(6.7,2.0),layout='constrained')
    for i,(loss,c) in enumerate(zip(LOSSES[:3],LC[:3])):
        rr=[r for r in ROWS if r['kind']=='joint_gradient_pair' and r['left']==loss and r['right']=='full_vocab']
        axs[0].scatter([i-.035,i+.035],[r['cosine'] for r in rr],color=c,s=20)
    axs[0].set(title='(a) Gradient',xticks=[0,1,2],xticklabels=['PG','I64','T64'],ylabel='Cosine with full',ylim=(.35,1.04),xlim=(-.5,2.5))
    direct=json.loads((DATA/'raw/direct_updates_results.json').read_text())
    for j,(q,c) in enumerate([('master','#0072B2'),('bf16','#D55E00')]):
        for i,loss in enumerate(LOSSES[:2]):
            vals=[r['cosine'] for bank in direct for r in bank['pairs'] if r['quantity']==q and r['left']==loss and r['right']=='full_vocab']
            axs[1].scatter([i+(j-.5)*.16]*len(vals),vals,c=c,marker='o' if j==0 else 's',s=20,label='FP32' if j==0 and i==0 else ('BF16' if i==0 else None))
    axs[1].set(title='(b) Adam step',xticks=[0,1],xticklabels=['PG','I64'],ylabel='Cosine with full',ylim=(.35,1.04),xlim=(-.5,1.5))
    axs[1].legend(frameon=False,loc='lower right')
    mc=json.loads((DATA/'raw/pg_variance_results.json').read_text())
    for bank,c in zip(mc,['#0072B2','#D55E00']):
        for repeat,marker in [(0,'o'),(1,'^')]:
            rr=sorted([r for r in bank['comparisons'] if r['repeat']==repeat],key=lambda r:r['draws'])
            axs[2].plot([r['draws'] for r in rr],[r['cosine'] for r in rr],color=c,marker=marker,alpha=.75,lw=.8,ms=3)
    axs[2].set(title='(c) PG action resampling',xscale='log',xticks=[1,16,64],xticklabels=['1','16','64'],xlabel='Actions per prefix',ylabel='Cosine with full',ylim=(.35,1.04))
    finish(fig);save(fig,'supervision_density',['raw/measurements.jsonl','raw/direct_updates_results.json','raw/pg_variance_results.json'])


def rollout_records(rows):
    records={}
    for r in rows:
        if 'rollout/step' in r['metrics']:
            records.setdefault(int(r['metrics']['rollout/step']),{}).update(r['metrics'])
    return records


def dynamics():
    rows=[json.loads(l) for l in (DATA/'raw/M-PG_rollout.jsonl').read_text().splitlines()]
    records=rollout_records(rows)
    fig,axs=plt.subplots(2,2,figsize=(6.7,3.55),layout='constrained')
    for ax,key,title,scale in zip(axs.flat,['token_share','mean_response_length','truncation_rate','terminal_advantage_mean'],
                                ['Token allocation','Response length','Truncated responses','Signal at completed terminal token'],[100,1,100,1]):
        for task,d,c in zip(TASK,['Math','Code','IF','Science'],DC):
            pairs=[(k,v[f'mopd/task/{task}/{key}']*scale) for k,v in sorted(records.items()) if f'mopd/task/{task}/{key}' in v]
            ax.plot([p[0] for p in pairs],[p[1] for p in pairs],color=c,label=d,lw=.7)
        ax.set(title=title,xlabel='Rollout index (before update)')
        if key=='token_share':ax.axhline(25,color='#777777',ls=':',lw=.8);ax.set_ylabel('Valid token share (%)')
        elif key=='mean_response_length':ax.set_ylabel('Tokens per response')
        elif key=='truncation_rate':ax.set_ylabel('Responses (%)')
        else:ax.set_ylabel('Mean log q − log p (nats)')
    axs[0,0].legend(frameon=False,ncol=2,loc='upper right');finish(fig)
    save(fig,'training_dynamics',['raw/M-PG_rollout.jsonl'])
    # Paired per-run token histories provide denominator context without imputing missing reward.
    fig,axs=plt.subplots(1,3,figsize=(6.7,2.05),layout='constrained')
    for ax,m in zip(axs.flat,['M-I64-DR','M-I64-DT','M-I64-GT']):
        rr=[json.loads(l) for l in (DATA/f'raw/{m}_rollout.jsonl').read_text().splitlines()]
        rec=rollout_records(rr)
        for task,d,c in zip(TASK,['Math','Code','IF','Science'],DC):
            xy=[(k,v[f'mopd/task/{task}/token_share']*100) for k,v in sorted(rec.items()) if f'mopd/task/{task}/token_share' in v]
            ax.plot(*zip(*xy),color=c,lw=.65,label=d)
        ax.axhline(25,color='#777777',ls=':',lw=.8);ax.set(title=m,xlabel='Rollout index',ylabel='Valid token share (%)',ylim=(0,80))
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
    finish(fig);save(fig,'all_token_shares',[f'raw/{m}_rollout.jsonl' for m in ['M-I64-DR','M-I64-DT','M-I64-GT']])


def supplemental():
    # Teacher JS is paired with the same bank, loss and teacher pair; no pooled regression.
    fig,axs=plt.subplots(2,2,figsize=(6.7,3.5),layout='constrained')
    pairkeys=sorted({(r['left'],r['right']) for r in ROWS if r['kind']=='teacher_js'})
    palette=plt.get_cmap('tab10')
    for ax,(loss,cond) in zip(axs.flat,[(l,c) for l in LOSSES[:2] for c in ['routed','common']]):
        for r in ROWS:
            if r['kind']!='teacher_pair' or r['loss']!=loss or r['condition']!=cond:continue
            j=next(j for j in ROWS if j['kind']=='teacher_js' and j['bank']==r['bank'] and j['left']==r['left'] and j['right']==r['right'])
            ax.scatter(j['mean'],1-r['support']['0.01']['jaccard'],color=palette(pairkeys.index((r['left'],r['right']))),
                       marker='o' if r['bank']==0 else '^',s=22)
        ax.set(title=('PG' if loss=='sampled_pg' else 'I64')+' / '+cond,xlabel='Teacher JS (nats)',ylabel='1 − gradient top-1% Jaccard',ylim=(0,1))
    fig.legend([Line2D([0],[0],marker='o',ls='',color=palette(i)) for i in range(6)],
               [a+' / '+b for a,b in pairkeys],loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    finish(fig);save(fig,'teacher_js',['raw/measurements.jsonl'])
    # Scale-aware threshold sensitivity for the local simulated writeback.
    fig,axs=plt.subplots(1,2,figsize=(6.7,2.1),layout='constrained')
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
    axs[0].legend(frameon=False,ncol=3,fontsize=7.5);finish(fig);save(fig,'thresholds',['raw/measurements.jsonl'])
    # Scores are accompanied by generation diagnostics rather than redefined by them.
    fig,axs=plt.subplots(1,3,figsize=(6.7,2.1),layout='constrained')
    rr=sorted([r for r in CAP if r['step']==100 or r['model']=='Initial'],key=lambda r:r['model'])
    for ax,key,title in zip(axs,['truncation','repetition','healthy_reward'],['IF truncation','IF repetition flag','IF completed and\nnonrepetitive reward']):
        vals=[100*r[key]['IF'] if r[key].get('IF') is not None else np.nan for r in rr]
        ax.scatter(range(len(rr)),vals,color=[COL.get(r['model'],'#777777') for r in rr],s=22)
        ax.set(xticks=range(len(rr)),xticklabels=[r['model'].replace('M-I64-','I64-') for r in rr],title=title,ylabel='All responses (%)')
        ax.tick_params(axis='x',rotation=50)
    finish(fig);save(fig,'if_generation',['capability.json'])


def tables():
    def write(name,lines):
        spec,header={
            'capability_rows.tex':('lrrrrr',r'Configuration & Updates & Math & Code & IF & GPQA'),
            'normalization_rows.tex':('lrrrr',r'I64 reduction, update 50 & Math & Code & IF & GPQA'),
            'teacher_rows.tex':('lrrr',r'Teacher / target domain & Responses & Score (\%) & Truncation (\%)'),
            'paired_rows.tex':('llrl',r'Reduction & Domain & Difference (pp) & 95\% interval'),
            'coverage_rows.tex':('rlrrrr',r'Bank & Domain & Mass in $T$ (\%) & Mass in $I$ (\%) & Entropy & $\|g_{\mathrm{PG}}\|_2$'),
        }[name]
        (DATA/name.replace('_rows','_table')).write_text('\n'.join(
            [r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule']+
            lines+[r'\bottomrule',r'\end{tabular}'])+'\n')
    order={'Initial':0,'S-PG':1,'S-I64':2,'M-PG':3,'M-I64-DR':4,'M-I64-DT':5,'M-I64-GT':6}
    rows=sorted(CAP,key=lambda r:(order[r['model']],r['step']))
    write('capability_rows.tex',[r['model']+' & '+str(r['step'])+' & '+' & '.join(f"{100*r['scores'][d]:.2f}" for d in DOM)+r' \\' for r in rows])
    write('normalization_rows.tex',[m.replace('M-I64-','')+' & '+' & '.join(f"{100*next(r for r in CAP if r['model']==m and r['step']==50)['scores'][d]:.2f}" for d in DOM)+r' \\' for m in ['M-I64-DR','M-I64-DT','M-I64-GT']])
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
    values={'capability':CAP,'paired_comparisons':COMPS,'geometry':GEO,'mechanism':SUM}
    (DATA/'plotted_values.json').write_text(json.dumps(values,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    capability();normalization();geometry();adam();teacher();density();dynamics();supplemental();tables()
    (DATA/'figure_manifest.json').write_text(json.dumps({'figures':plots,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(f'Generated {len(plots)} figures as PDF, PNG and SVG, plus five data tables.')
