from pathlib import Path
import subprocess
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root=Path.cwd()
out=root/'output/half_page_mechanism'
out.mkdir(parents=True,exist_ok=True)
ref='origin/archive/before-compile-only-20260914'
commit=subprocess.check_output(['git','rev-parse',ref],text=True).strip()
sources=[]
def read_archive(name):
    blob=subprocess.check_output(['git','show',f'{ref}:experiments/optimizer_mediation_20260913/{name}'])
    sources.append({'revision':commit,'path':f'experiments/optimizer_mediation_20260913/{name}',
                    'sha256':hashlib.sha256(blob).hexdigest()})
    (out/name).write_bytes(blob)
    return json.loads(blob)
rows=read_archive('measurements.json')['records']
online=read_archive('online_clip_coefficients.json')['records']
assert len(rows)==6 and len(online)==1500
colors={'DR':'#D55E00','DT':'#009E73','GT':'#AA579F'}
pair_colors={'DR_DT':'#0072B2','DR_GT':'#D55E00','DT_GT':'#009E73'}
summary={'sources':sources,'online':{},'local':{},'notes':[
    'The online series contain one recorded training trajectory per averaging rule, 500 updates each; no across-seed uncertainty is inferred.',
    'Fixed-state probes replay I64 gradients at common M-PG checkpoints 100 and 250, with three cached batches each. These are distinct from the independently trained online trajectories.',
    'SGD unit-direction discrepancy is analytically equal to gradient unit-direction discrepancy for positive scalar learning rates and global clipping. The SGD reference is an identity, not an extra measurement.',
    'Reset-m ratios describe step norms, not percentages of additive update energy or explained performance.',
    'Clearing m retains v and the optimizer step counter; clearing m,v also retains the counter. These are local interventions, not retrained RMSProp or fresh-Adam models.',
    'Two diagnostic checkpoints do not establish a monotone temporal law.']}
for rule in colors:
    seq=sorted([r for r in online if r['branch']==rule],key=lambda r:r['update'])
    assert [r['update'] for r in seq]==list(range(1,501))
    for r in seq:
        assert abs(r['clip_coefficient']-min(1.,1./(r['aggregate_grad_norm']+1e-6)))<1e-6
    summary['online'][rule]={}
    for lo,hi in [(1,100),(101,250),(251,500),(1,500)]:
        selected=[r for r in seq if lo<=r['update']<=hi]
        summary['online'][rule][f'{lo}-{hi}']={
            'clip_fraction':float(np.mean([r['clip_coefficient']<1 for r in selected])),
            'gradient_norm_median':float(np.median([r['aggregate_grad_norm'] for r in selected]))}
for step in [100,250]:
    rr=[r for r in rows if r['snapshot_step']==step]
    summary['local'][step]={'direction':{},'norm_ratios':{}}
    for pair in pair_colors:
        g=np.array([r['raw_gradient_comparisons'][pair]['cosine'] for r in rr])
        u=np.array([r['stateful_adam_master_update_comparisons'][pair]['cosine'] for r in rr])
        ratios=np.sqrt((1-u)/(1-g))
        summary['local'][step]['direction'][pair]={'gradient_cosine_mean':float(g.mean()),'adam_cosine_mean':float(u.mean()),
            'unit_direction_distance_ratio':ratios.tolist(),'ratio_mean':float(ratios.mean())}
    for rule in colors:
        v={}
        for kind in ['adam_reset_m','adam_reset_m_v']:
            ratios=np.array([r['branches'][kind+'/'+rule]['master_update_l2']/r['branches']['adam_stateful/'+rule]['master_update_l2'] for r in rr])
            v[kind]={'ratios':ratios.tolist(),'mean':float(ratios.mean())}
        summary['local'][step]['norm_ratios'][rule]=v
(out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,
    'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(7.5,2.55))
fig.subplots_adjust(left=.075,right=.99,bottom=.25,top=.81,wspace=.46)
ax=axes[0]
for rule,color in colors.items():
    seq=sorted([r for r in online if r['branch']==rule],key=lambda r:r['update'])
    a=np.array([r['aggregate_grad_norm'] for r in seq]).reshape(20,25)
    x=np.arange(13,501,25)
    ax.fill_between(x,np.quantile(a,.25,axis=1),np.quantile(a,.75,axis=1),color=color,alpha=.14,lw=0)
    ax.plot(x,np.median(a,axis=1),color=color,lw=1.2,label=rule)
ax.axhline(1,color='#555555',ls='--',lw=.9)
ax.text(490,1.08,'clip threshold',ha='right',fontsize=6.2,color='#555555')
ax.set(yscale='log',ylim=(.15,50),xlim=(0,505),xticks=[0,100,250,500],
       xlabel='Training update',ylabel='Pre-clipping gradient norm')
ax.set_title('(a) Different clipping regimes',loc='left',pad=8)
ax.legend(loc='upper right',frameon=False,ncol=3,fontsize=6,handlelength=1.1,columnspacing=.7)

ax=axes[1]
for j,(pair,color) in enumerate(pair_colors.items()):
    for i,step in enumerate([100,250]):
        vals=np.array(summary['local'][step]['direction'][pair]['unit_direction_distance_ratio'])
        xpos=i+(j-1)*.15
        ax.scatter(xpos+np.array([-.025,0,.025]),vals,s=13,facecolors='white',edgecolors=color,lw=.7,zorder=3)
        ax.plot(xpos,vals.mean(),marker='_',ms=9,color=color,mew=1.5,zorder=4)
    ax.plot([0+(j-1)*.15,1+(j-1)*.15],[summary['local'][step]['direction'][pair]['ratio_mean'] for step in [100,250]],color=color,lw=.8,alpha=.7)
ax.axhline(1,color='#555555',ls='--',lw=.9)
ax.text(.5,1.025,'SGD / clipping: 1 (identity)',ha='center',fontsize=6.2,color='#555555')
ax.set(xlim=(-.45,1.45),ylim=(0,1.15),xticks=[0,1],xticklabels=['100','250'],
       xlabel='Diagnostic checkpoint',ylabel='Adam / gradient direction distance')
ax.set_title('(b) Direction attenuation',loc='left',pad=8)
ax.legend(handles=[Line2D([0],[0],color=c,lw=1,label=p.replace('_','-')) for p,c in pair_colors.items()],
          loc='upper center',bbox_to_anchor=(.5,-.28),ncol=3,frameon=False,fontsize=6,handlelength=.8,columnspacing=.65)

ax=axes[2]
for j,(rule,color) in enumerate(colors.items()):
    for kind,ls,marker in [('adam_reset_m','-','o'),('adam_reset_m_v',':','s')]:
        means=[]
        for i,step in enumerate([100,250]):
            vals=np.array(summary['local'][step]['norm_ratios'][rule][kind]['ratios'])
            xpos=i+(j-1)*.13
            ax.scatter(xpos+np.array([-.02,0,.02]),vals,s=12,marker=marker,facecolors='white',edgecolors=color,lw=.65,zorder=3)
            ax.plot(xpos,vals.mean(),marker='_',ms=8,color=color,mew=1.3,zorder=4)
            means.append(vals.mean())
        ax.plot([0+(j-1)*.13,1+(j-1)*.13],means,ls=ls,color=color,lw=.9,alpha=.75)
ax.axhline(1,color='#777777',lw=.75)
ax.text(.55,1.08,'saved state',fontsize=6.2,color='#777777')
ax.set(xlim=(-.4,1.45),ylim=(0,3.25),xticks=[0,1],xticklabels=['100','250'],
       xlabel='Diagnostic checkpoint',ylabel='Intervened / saved step norm')
ax.set_title('(c) Moment interventions',loc='left',pad=8)
ax.legend(handles=[Line2D([0],[0],color='#444444',ls='-',marker='o',ms=3,label='Reset m'),
                   Line2D([0],[0],color='#444444',ls=':',marker='s',ms=3,label='Reset m,v')],
          loc='upper center',bbox_to_anchor=(.5,-.28),ncol=2,frameon=False,fontsize=6,handlelength=1.5,columnspacing=1.)
for ax in axes:
    ax.grid(axis='y',alpha=.2,lw=.5)
    ax.set_axisbelow(True)
fig.text(.078,.965,'Normalization changes the signal entering the optimizer; Adam changes how that signal becomes a step.',
         fontsize=8.1,fontweight='bold',va='top')
fig.savefig(out/'normalization_optimizer_mechanism.png',dpi=260)
fig.savefig(out/'normalization_optimizer_mechanism.svg')
print(json.dumps({'output':str(out),'archive_commit':commit,'counts':[len(rows),len(online)],'status':'verified'}))
