"""Repeated fresh-action Monte Carlo gradients on exactly the prepared prefixes."""
import fcntl
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM

from prepare import ROOT,CONFIG,prefix_logits,digest
from measure import gram,pair_from_gram,local_distillation_loss

OUT=ROOT/'pg_variance'


def write(name,data):
    p=OUT/name;tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def status(stage,**kw):
    d={'stage':stage,'time':time.time(),'pid':os.getpid(),**kw};write('status.json',d);print(json.dumps(d),flush=True)


def gradient(model,rows,scores,bank,draws,seed,names,sizes):
    model.zero_grad(set_to_none=True)
    action_rng=torch.Generator().manual_seed(seed)
    selected=[]
    for ri,row in enumerate(rows):
        logits=prefix_logits(model,row)
        logp=logits.log_softmax(-1)
        q=scores[row['task']][bank][ri].to(logp)
        if draws is None:
            loss=local_distillation_loss(logits,q,loss='full_vocab',weights=torch.full((len(row['positions']),),.25/len(row['positions']),device='cuda'))
        else:
            ids=torch.multinomial(row['student_log_probs'].exp(),draws,replacement=True,generator=action_rng).to('cuda')
            lp=logp.gather(-1,ids)
            advantage=(q.gather(-1,ids)-lp).detach()
            loss=-(advantage*lp).mean()*.25
            selected.append(ids.cpu().tolist())
        assert torch.isfinite(loss)
        loss.backward()
    out=torch.empty(sum(sizes),dtype=torch.float32)
    params=dict(model.named_parameters());offset=0
    for n,size in zip(names,sizes):
        assert params[n].grad is not None
        out[offset:offset+size].copy_(params[n].grad.detach().reshape(-1).cpu())
        offset+=size
    model.zero_grad(set_to_none=True)
    return out,selected


def main():
    started=time.time()
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(3);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    lock=(OUT/'gpu6.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    assert digest(ROOT/'banks.pt')==manifest['banks_sha256']
    assert digest(ROOT/'teacher_scores.pt')==manifest['teacher_scores_sha256']
    rows=torch.load(ROOT/'banks.pt',weights_only=True);scores=torch.load(ROOT/'teacher_scores.pt',weights_only=True)
    status('load_student')
    model=AutoModelForCausalLM.from_pretrained(CONFIG['hf_checkpoint'],dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    names=sorted(dict(model.named_parameters()));sizes=[dict(model.named_parameters())[n].numel() for n in names]
    assert sum(sizes)==1720574976
    result=[];action_records=[]
    for bank,bank_rows in enumerate(rows):
        vectors=[];labels=[]
        status('full_vocab',bank=bank)
        ref,_=gradient(model,bank_rows,scores,bank,None,0,names,sizes)
        vectors.append(ref);labels.append({'loss':'full_vocab'})
        for draws in [1,16,64]:
            for repeat in range(2):
                seed=20260910+bank*100003+draws*1009+repeat
                status('sampled_pg',bank=bank,draws=draws,repeat=repeat)
                g,actions=gradient(model,bank_rows,scores,bank,draws,seed,names,sizes)
                vectors.append(g);labels.append({'loss':'sampled_pg','draws':draws,'repeat':repeat,'seed':seed})
                action_records.append({'bank':bank,**labels[-1],'actions':actions})
                write('actions.json',action_records)
        status('gram',bank=bank)
        matrix=gram(vectors)
        comparisons=[]
        for i in range(1,len(vectors)):
            pair=pair_from_gram(matrix,i,0)
            comparisons.append({**labels[i],**pair,'relative_gradient_error':pair['difference_l2']/pair['right_l2']})
        result.append({'bank':bank,'labels':labels,'gram':matrix.tolist(),'comparisons':comparisons})
        write('results.json',result)
        del vectors,ref,g
    write('run_complete.json',{'status':'complete','elapsed_seconds':time.time()-started,
        'banks':2,'repeats_per_draw_count':2,'draw_counts':[1,16,64],
        'results_sha256':digest(OUT/'results.json'),'actions_sha256':digest(OUT/'actions.json'),
        'source_sha256':digest(Path(__file__)),'prefixes_and_teacher_scores_shared_with_main_probe':True,
        'scope':'Conditional sampling variability at fixed prefixes; not training seed variation',
        'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated()})
    status('complete')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():status('failed',error=repr(exc))
        raise
