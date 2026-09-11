#!/usr/bin/env python3
"""Render three exploratory triptychs from the frozen audit, without missing-data imputation."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DOMAINS=['math','code','if','science']
NAMES=['Math','Code','IF','Science']
COLORS=['#2563a6','#bd5b32','#24816c','#8555a6']
DATASETS=['math500_pass1','livecodebench_postcutoff_pass1','ifbench_strict','gpqa_diamond_avg4']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180})

def clean(ax):
    ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)

def save(fig,path):
    fig.savefig(path.with_suffix('.png'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path(__file__).parent/'figure_audit_20260908/evidence_snapshot.json')
    ap.add_argument('--output',type=Path,default=Path(__file__).parent.parent/'figures/evidence_audit_20260908')
    a=ap.parse_args(); data=json.loads(a.input.read_text());a.output.mkdir(parents=True,exist_ok=True)
    figures=[]
    fig,axes=plt.subplots(1,3,figsize=(13.6,3.6),layout='constrained')
    figures.append(fig)
    ax=axes[0]
    by_key={(r['rollout_id'],r['domain']):r for r in data['rollout_domain_metrics'] if r['run_id']=='m-pg-s42'}
    steps=sorted({x[0] for x in by_key})
    for d,name,c in zip(DOMAINS,NAMES,COLORS):
        xs,ys=[],[]
        for start in range(0,len(steps),25):
            batch=steps[start:start+25]
            den=sum(by_key[(s,t)]['valid_response_tokens'] for s in batch for t in DOMAINS)
            xs.append(batch[-1]+1);ys.append(100*sum(by_key[(s,d)]['valid_response_tokens'] for s in batch)/den)
        ax.plot(xs,ys,color=c,label=name,lw=1.8)
    ax.axhline(25,color='#777',ls=':',lw=1,label='Prompt share (each)')
    ax.set(title='F1b  Equal prompts, unequal token shares',xlabel='M-PG optimizer updates',ylabel='Valid response tokens (%)')
    ax.legend(ncol=2);clean(ax)
    ax=axes[1]
    n=next(r for r in data['normalization_probes'] if r['loss']=='topk_intersection')
    methods=['zero_gradient','DR','DT','GT']; labels=['Zero grad.','DR','DT','GT']
    vals=np.array([[n['branches'][m]['heldout_kl_decrease'][d] for d in DOMAINS] for m in methods])
    vmax=np.max(np.abs(vals));im=ax.imshow(vals,cmap='RdBu',vmin=-vmax,vmax=vmax,aspect='auto')
    ax.set_xticks(range(4),NAMES);ax.set_yticks(range(4),labels)
    for i in range(4):
        for k in range(4):ax.text(k,i,f'{vals[i,k]:+.4f}',ha='center',va='center',fontsize=8,color='white' if abs(vals[i,k])>.65*vmax else '#222')
    ax.set_title('F2c  Local held-out KL decrease')
    ax.set_xlabel('M-PG/500 state; intersection loss; cap 512')
    fig.colorbar(im,ax=ax,shrink=.8,label='KL before - after (nats)')
    ax=axes[2]
    ev={}
    # Preserve the first verified complete attempt; never select by score.
    for r in sorted(data['evaluations'],key=lambda x:x['completed_at_utc']):
        if r['verified']: ev.setdefault((r['family'],r['step'],r['dataset']),r)
    for ds,name,c in zip(DATASETS,NAMES,COLORS):
        vals=[]
        for t in [0,100,250,500]:
            r=ev.get(('initial_student' if t==0 else 'M-PG',t,ds))
            vals.append(r['score']*100 if r else np.nan)
        ax.plot([0,100,250,500],vals,'o-',label=name,color=c,lw=1.8,ms=4)
    ax.set(title='F8b (partial)  M-PG capability',xlabel='Optimizer updates',ylabel='Benchmark score (%)')
    ax.legend(ncol=2);clean(ax)
    fig.suptitle('Exploratory evidence | training seed 42 | F1b: 25-update token totals; F2c: 8 train / 4 held-out responses; F8b: IF/100 missing',fontsize=10)
    save(fig,a.output/'01_balancing_and_capability')

    fig,axes=plt.subplots(1,3,figsize=(13.6,3.6),layout='constrained');figures.append(fig)
    ids=['s-pg-s42','s-tk64-s42','m-pg-s42','m-tk64-dr-s42','m-tk64-dt-s42','m-tk64-gt-s42']
    labels=['S-PG','S-ST64','M-PG','M-ST64-DR','M-ST64-DT','M-ST64-GT']
    cs=['#6d8295','#8555a6','#2463a6','#24816c','#bd5b32','#cb9a27']
    ax=axes[0]
    for rid,label,c in zip(ids,labels,cs):
        rows=sorted([r for r in data['bf16_scans'] if r['actual_checkpoint_run']==rid and r['step']<=100],key=lambda r:r['step'])
        ax.plot([0]+[r['step'] for r in rows],[0]+[100*r['global']['effective_changed_fraction'] for r in rows],
                marker='o',lw=1.7,ms=3,color=c,label=label,ls='--' if rid.startswith('s-') else '-')
    ax.set(title='F4a (partial)  Cumulative BF16 changes',xlabel='Optimizer updates',ylabel='Parameters with |delta| > 1e-5 (%)')
    ax.legend(ncol=2);clean(ax)
    ax=axes[1]
    layers=['embedding_and_head']+[f'model.layers.{i}' for i in range(28)]
    rows=[next(r for r in data['bf16_scans'] if r['actual_checkpoint_run']==rid and r['step']==100) for rid in ids]
    vals=np.array([[100*r['layers'][l]['effective_changed_fraction'] for l in layers] for r in rows])
    im=ax.imshow(vals,aspect='auto',vmin=0,cmap='YlGnBu')
    ax.set_yticks(range(6),labels);ax.set_xticks([0,8,15,22,28],['Emb.','7','14','21','27'])
    ax.set(title='F4c (partial)  Layer concentration at 100',xlabel='Transformer layer (Emb. includes head)')
    fig.colorbar(im,ax=ax,shrink=.8,label='Changed parameters (%)')
    ax=axes[2]; x=np.arange(2);width=.19
    for i,(branch,label,c) in enumerate(zip(['zero_gradient','sampled_pg','student_topk','full_vocab'],
                                          ['Zero grad.','PG','Student Top64','Full vocab.'],['#9b9b9b','#2463a6','#24816c','#bd5b32'])):
        vals=[]
        for prefix in ['s-pg250','m-pg250']:
            p=next(p for p in data['local_probes'] if p['job'].startswith(prefix))
            vals.append(100*p['branches'][branch]['metrics']['bf16_writeback']['fraction_above_1e-05'])
        ax.bar(x+(i-1.5)*width,vals,width,label=label,color=c)
    ax.set_xticks(x,['S-PG/250','M-PG/250'])
    ax.set(title='F7a (partial)  Local BF16 writeback',ylabel='Parameters with |update| > 1e-5 (%)')
    ax.set_ylim(0,.068);ax.legend(ncol=2,loc='upper center');clean(ax)
    fig.suptitle('Exploratory evidence | PG / student-support Top64 checkpoints | Single / multi domain exposure differs | Local probe: cap 128, 2 prefixes/response',fontsize=10)
    save(fig,a.output/'02_sparsity_and_density')

    fig,axes=plt.subplots(1,3,figsize=(13.6,3.6),layout='constrained');figures.append(fig)
    pairs=[r for r in data['teacher_pairs'] if r['threshold']==1e-5]
    ax=axes[0];mat=np.eye(4)
    for i,d in enumerate(DOMAINS):
        for k,e in enumerate(DOMAINS):
            if i==k:continue
            rows=[r for r in pairs if r['mode']=='routed' and {r['teacher_left'],r['teacher_right']}=={d,e}]
            mat[i,k]=np.mean([r['bf16_jaccard'] for r in rows])
    im=ax.imshow(mat,cmap='YlGnBu',vmin=0,vmax=1)
    ax.set_xticks(range(4),NAMES);ax.set_yticks(range(4),NAMES)
    for i in range(4):
        for k in range(4):ax.text(k,i,f'{mat[i,k]:.3f}',ha='center',va='center',color='white' if mat[i,k]>.55 else '#222',fontsize=9)
    ax.set_title('F5a (partial)  Routed teacher overlap')
    fig.colorbar(im,ax=ax,shrink=.8,label='BF16 Jaccard, |update| > 1e-5')
    ax=axes[1];x=np.arange(4)
    for offset,mode,c in [(-.12,'routed','#2463a6'),(.12,'common','#bd5b32')]:
        vals=[]
        for task in DOMAINS:
            observations=[]
            for p in data['local_probes']:
                if 'draw' not in p['job']:continue
                observations.extend(r['metrics']['support']['1e-05']['jaccard'] for r in p['pairs']
                    if r['left']=='zero_gradient' and r['right']==f'{mode}/{task}/sampled_pg')
            vals.append(observations)
        means=np.mean(vals,axis=1);lo=np.min(vals,axis=1);hi=np.max(vals,axis=1)
        ax.errorbar(x+offset,means,yerr=[means-lo,hi-means],fmt='o',capsize=3,color=c,label=mode)
    ax.set_xticks(x,NAMES);ax.set_ylim(0,1.04)
    ax.set(title='F5c  Overlap with zero-gradient step',ylabel='BF16 Jaccard')
    ax.legend();clean(ax)
    ax=axes[2]
    pair_ids=[(d,e) for i,d in enumerate(DOMAINS) for e in DOMAINS[i+1:]]
    pair_cs=['#2463a6','#bd5b32','#24816c','#8555a6','#b58c20','#6e7780']
    for (d,e),c in zip(pair_ids,pair_cs):
        for mode,marker in [('routed','o'),('common','^')]:
            rs=[r for r in pairs if r['mode']==mode and (r['teacher_left'],r['teacher_right'])==(d,e)]
            xs=np.array([r['js_common_prefixes'] for r in rs]);ys=np.array([r['bf16_distance'] for r in rs])
            ax.scatter(xs,ys,color=c,marker=marker,s=15,alpha=.2)
            ax.scatter(xs.mean(),ys.mean(),color=c,marker=marker,s=45)
            if mode=='routed':ax.annotate(f'{d[0].upper()}-{e[0].upper()}',(xs.mean(),ys.mean()),xytext=(3,3),textcoords='offset points',fontsize=7,color=c)
    for mode,marker in [('Routed mean','o'),('Common mean','^')]:ax.scatter([],[],color='#444',marker=marker,label=mode)
    ax.set(title='F6b/c (partial)  Teacher JS vs support distance',xlabel='Teacher JS on common prefixes (nats)',ylabel='1 - BF16 Jaccard')
    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.set_ylim(bottom=0);ax.legend();clean(ax)
    fig.suptitle('M-PG/500 only | one training seed, three diagnostic banks (42/43/44) | Bars: observed min-max, not confidence intervals | Shared teacher pairs are dependent',fontsize=10)
    save(fig,a.output/'03_teacher_overlap')
    with PdfPages(a.output/'exploratory_triptychs.pdf') as pdf:
        for fig in figures:pdf.savefig(fig,bbox_inches='tight')
    for fig in figures:plt.close(fig)
    (a.output/'preview_manifest.json').write_text(json.dumps({'input':str(a.input),
        'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
        'plotter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evaluation_repeat_selection':'earliest verified complete attempt, not highest score','audit_finished_utc':data['audit_finished_utc'],
        'panel_ids':['F1b','F2c','F8b partial','F4a partial','F4c partial','F7a partial','F5a partial','F5c','F6b/c partial'],
        'scope':'Exploratory previews, not completed 24-panel evidence; no data imputation or fitted teacher-pair trend',
        'source_labels':'Method labels follow checkpoint provenance.'},indent=2)+'\n')
    print('Rendered three 1x3 exploratory rows and a three-page PDF:',a.output)
if __name__=='__main__':main()
