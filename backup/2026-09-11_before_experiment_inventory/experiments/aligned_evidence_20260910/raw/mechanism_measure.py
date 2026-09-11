import fcntl
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from prepare import ROOT, CONFIG, TASKS, prefix_logits, digest, write

sys.path.insert(0, str(ROOT.parents[1]))
from slime_plugins.mopd.paper_diagnostics import local_distillation_loss, js_divergence

CHUNK = 2_000_000
THRESHOLDS = [0., 1e-8, 1e-7, 1e-6, 1e-5]


def status(stage, **kw):
    d = {'stage':stage, 'time':time.time(), 'pid':os.getpid(), **kw}
    write('measure_status.json',d)
    print(json.dumps(d),flush=True)


def emit(kind, **kw):
    with (ROOT/'measurements.jsonl').open('a') as f:
        f.write(json.dumps({'kind':kind, **kw},allow_nan=False)+'\n')


def gram(vectors, device='cuda'):
    """Full-coordinate FP64 dot products; no projection or parameter sampling."""
    result = torch.zeros((len(vectors),len(vectors)),dtype=torch.float64,device=device)
    for start in range(0, vectors[0].numel(), CHUNK):
        block = torch.stack([v[start:start+CHUNK] for v in vectors]).to(device=device,dtype=torch.float64)
        result.add_(block @ block.T)
    return result.cpu().numpy()


def pair_from_gram(matrix, i, j):
    a,b,dot = float(matrix[i,i]),float(matrix[j,j]),float(matrix[i,j])
    return {'cosine': dot/math.sqrt(a*b) if a>0 and b>0 else None,
            'left_l2':math.sqrt(max(0,a)), 'right_l2':math.sqrt(max(0,b)),
            'difference_l2':math.sqrt(max(0,a+b-2*dot))}


def describe(vector, support=False):
    x=vector.numpy()
    energy=float(np.einsum('i,i->',x,x,dtype=np.float64))
    magnitude=np.abs(x)
    assert np.isfinite(magnitude).all()
    metrics={'parameters':len(x),'l2':math.sqrt(energy),
        'fraction_above':{str(t):float(np.count_nonzero(magnitude>t)/len(x)) for t in THRESHOLDS}}
    packed={}
    fractions=[0.001,0.01]
    sizes=[max(1,math.ceil(len(x)*f)) for f in fractions]
    magnitude.partition([len(x)-n for n in sizes])
    metrics['top_energy']={str(f):float(np.einsum('i,i->',magnitude[-n:],magnitude[-n:],dtype=np.float64)/energy) if energy else None
                           for f,n in zip(fractions,sizes)}
    if support:
        cutoffs=[float(magnitude[-n]) for n in sizes]
        del magnitude
        magnitude=np.abs(x)
        metrics['support']={}
        for fraction,count,cutoff in zip(fractions,sizes,cutoffs):
            mask=magnitude>cutoff
            needed=count-int(mask.sum()) if energy else 0
            equal_count=int(np.count_nonzero(magnitude==cutoff))
            remaining=needed
            for start in range(0,len(x),CHUNK):
                if remaining<=0: break
                equal=np.flatnonzero(magnitude[start:start+CHUNK]==cutoff)[:remaining]
                mask[start+equal]=True
                remaining-=len(equal)
            assert remaining==0
            packed[str(fraction)]=np.packbits(mask)
            metrics['support'][str(fraction)]={'count':int(mask.sum()),'cutoff':cutoff,
                'tie_dominated':equal_count>needed,'tie_rule':'stable_coordinate_index'}
    return metrics,packed


POPCOUNT=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)


def overlap(a,b,na,nb,population):
    intersection=int(POPCOUNT[np.bitwise_and(a,b)].sum(dtype=np.int64))
    union=na+nb-intersection
    expected=na*nb/population
    return {'jaccard':intersection/union if union else None,'intersection':intersection,
        'union':union,'random_jaccard':expected/(na+nb-expected) if na+nb>expected else None}


def collect(model,row,q,loss,names,shapes):
    model.zero_grad(set_to_none=True)
    logits=prefix_logits(model,row)
    objective=local_distillation_loss(logits,q.to(logits),loss=loss,
        weights=torch.full((len(row['positions']),),1/len(row['positions']),device='cuda'),
        action_ids=row['actions'],advantage_clip=0.,topk=64)
    assert torch.isfinite(objective)
    objective.backward()
    result=torch.empty(sum(shapes),dtype=torch.float32)
    params=dict(model.named_parameters())
    offset=0
    for name,size in zip(names,shapes):
        assert params[name].grad is not None, name
        result[offset:offset+size].copy_(params[name].grad.detach().reshape(-1).cpu())
        offset+=size
    model.zero_grad(set_to_none=True)
    return result,float(objective.detach())


def adam_chunk(state, gradient, mode='history'):
    """Local non-fused AdamW arithmetic; reset modes are optimizer interventions."""
    beta1,beta2=state['betas']
    step=int(state['step'])+1
    m,v=state['exp_avg'],state['exp_avg_sq']
    if mode=='fresh':
        m=torch.zeros_like(m); v=torch.zeros_like(v); step=1
    elif mode=='reset_m':
        m=torch.zeros_like(m)
    elif mode!='history':
        raise ValueError(mode)
    m=m*beta1+gradient*(1-beta1)
    v=v*beta2+gradient.square()*(1-beta2)
    if state.get('bias_correction',True):
        m=m/(1-beta1**step); v=v/(1-beta2**step)
    after=state['value']*(1-state['lr']*state['weight_decay'])-state['lr']*m/(v.sqrt()+state['eps'])
    return after


def adam(snapshot,names,gradient,mode='history'):
    count=sum(snapshot['parameters'][n]['value'].numel() for n in names)
    norm=math.sqrt(float(np.einsum('i,i->',gradient.numpy(),gradient.numpy(),dtype=np.float64))) if gradient is not None else 0.
    coefficient=min(1.,snapshot['clip_grad']/(norm+1e-6)) if snapshot['clip_grad']>0 else 1.
    master=torch.empty(count,dtype=torch.float32); bf16=torch.empty_like(master)
    offset=0
    for name in names:
        state=snapshot['parameters'][name]
        count_here=state['value'].numel()
        for start in range(0,count_here,CHUNK):
            stop=min(start+CHUNK,count_here); sl=slice(offset+start,offset+stop)
            local={k:(v.reshape(-1)[start:stop].to('cuda') if torch.is_tensor(v) else v) for k,v in state.items()}
            g=torch.zeros_like(local['value']) if gradient is None else gradient[sl].to('cuda')*coefficient
            after=adam_chunk(local,g,mode)
            master[sl].copy_((after-local['value']).cpu())
            bf16[sl].copy_((after.bfloat16().float()-local['model_value'].float()).cpu())
        offset+=count_here
    return master,bf16,{'gradient_l2':norm,'clip_coefficient':coefficient,'mode':mode}


def main():
    started=time.time()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    lock=(ROOT/'gpu4.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (ROOT/'prepare_complete.json').exists()
    if (ROOT/'measurements.jsonl').exists(): raise RuntimeError('Refuse to overwrite measurements')
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    assert digest(ROOT/'banks.pt')==manifest['banks_sha256']
    assert digest(ROOT/'teacher_scores.pt')==manifest['teacher_scores_sha256']
    banks=torch.load(ROOT/'banks.pt',weights_only=True)
    scores=torch.load(ROOT/'teacher_scores.pt',weights_only=True)
    status('load_snapshot')
    snapshot=torch.load(CONFIG['snapshot'],mmap=True,weights_only=True,map_location='cpu')
    names=sorted(snapshot['parameters'])
    shapes=[snapshot['parameters'][n]['value'].numel() for n in names]
    assert sum(shapes)==1720574976
    # No-update conversion checks whether the stored model was already rounded differently.
    rounding_mismatches=0
    for n in names:
        s=snapshot['parameters'][n]
        rounding_mismatches+=int((s['value'].bfloat16()!=s['model_value']).sum())
    emit('no_update_conversion',mismatched_coordinates=rounding_mismatches,parameters=sum(shapes))
    status('zero_gradient_adam')
    zero_m,zero_b,zero_meta=adam(snapshot,names,None)
    for q,v in [('master',zero_m),('bf16',zero_b)]:
        met,_=describe(v); emit('optimizer',bank='shared',loss='zero_gradient',quantity=q,metrics=met,**zero_meta)
    status('load_student')
    model=AutoModelForCausalLM.from_pretrained(CONFIG['hf_checkpoint'],dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    assert all(p.dtype==torch.float32 for p in model.parameters())
    for bank_id,rows in enumerate(banks):
        # Record full-vocabulary teacher differences on identical prefixes.
        for i,j in itertools.combinations(range(4),2):
            per_domain={r['task']:js_divergence(scores[TASKS[i]][bank_id][k],scores[TASKS[j]][bank_id][k]) for k,r in enumerate(rows)}
            emit('teacher_js',bank=bank_id,left=TASKS[i],right=TASKS[j],per_domain=per_domain,mean=sum(per_domain.values())/4)
        joint={}
        for loss in ['sampled_pg','topk_intersection']:
            grid=[]
            labels=[]
            for teacher in TASKS:
                for k,row in enumerate(rows):
                    status('gradient',bank=bank_id,loss=loss,teacher=teacher,input_domain=row['task'])
                    g,obj=collect(model,row,scores[teacher][bank_id][k],loss,names,shapes)
                    grid.append(g); labels.append({'teacher':teacher,'input_domain':row['task']})
                    emit('objective',bank=bank_id,loss=loss,teacher=teacher,input_domain=row['task'],surrogate=obj)
            status('crossed_gram',bank=bank_id,loss=loss)
            mat=gram(grid)
            emit('crossed_gram',bank=bank_id,loss=loss,labels=labels,gram=mat.tolist())
            for i,j in itertools.combinations(range(16),2):
                same_t=labels[i]['teacher']==labels[j]['teacher']
                same_d=labels[i]['input_domain']==labels[j]['input_domain']
                if same_t or same_d:
                    emit('crossed_pair',bank=bank_id,loss=loss,left=labels[i],right=labels[j],
                        condition='same_teacher_different_input' if same_t else 'same_input_different_teacher',**pair_from_gram(mat,i,j))
            joint[loss]=sum((grid[i*4+i]*.25 for i in range(4)),torch.zeros_like(grid[0]))
            common=[sum((grid[i*4+j]*.25 for j in range(4)),torch.zeros_like(grid[0])) for i in range(4)]
            routed=[grid[i*4+i] for i in range(4)]
            for condition,vectors in [('common',common),('routed',routed)]:
                matrix=gram(vectors)
                supports=[]; metrics=[]
                for teacher,vector in zip(TASKS,vectors):
                    status('gradient_support',bank=bank_id,loss=loss,condition=condition,teacher=teacher)
                    met,sup=describe(vector,support=True); metrics.append(met); supports.append(sup)
                    emit('gradient_metrics',bank=bank_id,loss=loss,condition=condition,teacher=teacher,metrics=met)
                for i,j in itertools.combinations(range(4),2):
                    overlap_metrics={f:overlap(supports[i][f],supports[j][f],metrics[i]['support'][f]['count'],
                        metrics[j]['support'][f]['count'],sum(shapes)) for f in supports[i]}
                    emit('teacher_pair',bank=bank_id,loss=loss,condition=condition,left=TASKS[i],right=TASKS[j],
                        **pair_from_gram(matrix,i,j),support=overlap_metrics)
                del supports,metrics
            del grid,common,routed,g,vectors,vector
            gc.collect()
        for loss in ['teacher_topk','full_vocab']:
            combined=torch.zeros(sum(shapes),dtype=torch.float32)
            for k,row in enumerate(rows):
                status('reference_gradient',bank=bank_id,loss=loss,input_domain=row['task'])
                g,obj=collect(model,row,scores[row['task']][bank_id][k],loss,names,shapes)
                combined.add_(g,alpha=.25)
                del g
            joint[loss]=combined
        loss_names=list(joint)
        matrix=gram(list(joint.values()))
        emit('joint_gradient_gram',bank=bank_id,losses=loss_names,gram=matrix.tolist())
        for i,j in itertools.combinations(range(len(loss_names)),2):
            emit('joint_gradient_pair',bank=bank_id,left=loss_names[i],right=loss_names[j],**pair_from_gram(matrix,i,j))
        for loss,g in joint.items():
            gm,_=describe(g); emit('joint_gradient',bank=bank_id,loss=loss,metrics=gm)
            for mode in ['history','fresh','reset_m']:
                if mode=='reset_m' and loss not in ['sampled_pg','topk_intersection']: continue
                status('adam',bank=bank_id,loss=loss,mode=mode)
                m,b,meta=adam(snapshot,names,g,mode)
                for quantity,v,zero in [('master',m,zero_m),('bf16',b,zero_b)]:
                    met,_=describe(v)
                    emit('optimizer',bank=bank_id,loss=loss,quantity=quantity,metrics=met,**meta)
                    if mode=='history':
                        pair=gram([v,zero])
                        residual=v-zero
                        rm,_=describe(residual)
                        emit('optimizer_zero_comparison',bank=bank_id,loss=loss,quantity=quantity,
                            **pair_from_gram(pair,0,1),residual=rm)
                        del residual
                del m,b,v,zero
            gc.collect()
        del joint,g,combined
        gc.collect()
        write(f'bank_{bank_id}_complete.json',{'status':'complete','bank':bank_id,'time':time.time()})
    source_hashes={p.name:digest(p) for p in ROOT.glob('*.py')}
    write('measure_complete.json',{'status':'complete','elapsed_seconds':time.time()-started,
        'measurement_sha256':digest(ROOT/'measurements.jsonl'),'source_sha256':source_hashes,
        'snapshot_step':100,'banks':len(banks),'parameters':sum(shapes),'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
        'precision':'FP32 local gradients at stored BF16 model weights; non-fused FP32 Adam simulation; TF32 disabled',
        'scope':'local counterfactual; fresh resets m,v,step; reset_m retains v and step; not online update or teacher attribution'})
    status('complete')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        status('failed',error=repr(exc))
        raise
