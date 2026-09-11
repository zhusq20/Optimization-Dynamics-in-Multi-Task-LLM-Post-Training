"""Direct pairwise step directions, supplementing the streamed marginal metrics."""
import fcntl
import itertools
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM

from prepare import ROOT,CONFIG,digest
from measure import collect,adam,gram,pair_from_gram

OUT=ROOT/'direct_updates'


def write(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)


def status(stage,**kw):
    d={'stage':stage,'time':time.time(),'pid':os.getpid(),**kw};write('status.json',d);print(json.dumps(d),flush=True)


def main():
    started=time.time();OUT.mkdir(exist_ok=False)
    torch.set_num_threads(3);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    lock=(OUT/'gpu6.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    assert digest(ROOT/'banks.pt')==manifest['banks_sha256']
    assert digest(ROOT/'teacher_scores.pt')==manifest['teacher_scores_sha256']
    banks=torch.load(ROOT/'banks.pt',weights_only=True);scores=torch.load(ROOT/'teacher_scores.pt',weights_only=True)
    status('load_snapshot')
    snapshot=torch.load(CONFIG['snapshot'],mmap=True,weights_only=True,map_location='cpu')
    assert snapshot['step']==100
    names=sorted(snapshot['parameters']);sizes=[snapshot['parameters'][n]['value'].numel() for n in names]
    status('zero_adam')
    zm,zb,_=adam(snapshot,names,None)
    status('load_student')
    model=AutoModelForCausalLM.from_pretrained(CONFIG['hf_checkpoint'],dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    results=[]
    for bank,rows in enumerate(banks):
        master=[zm];bf16=[zb];labels=['zero_gradient'];norms={}
        for loss in ['sampled_pg','topk_intersection','full_vocab']:
            gs=[]
            for ri,row in enumerate(rows):
                status('gradient',bank=bank,loss=loss,domain=row['task'])
                g,_=collect(model,row,scores[row['task']][bank][ri],loss,names,sizes);gs.append(g)
            joint=sum((g*.25 for g in gs),torch.zeros_like(gs[0]))
            del gs,g
            status('adam',bank=bank,loss=loss)
            m,b,meta=adam(snapshot,names,joint)
            norms[loss]=meta['gradient_l2'];master.append(m);bf16.append(b);labels.append(loss)
            del joint,m,b
        status('gram',bank=bank)
        entry={'bank':bank,'labels':labels,'gradient_l2':norms,'pairs':[]}
        for quantity,vectors in [('master',master),('bf16',bf16)]:
            matrix=gram(vectors)
            entry[quantity+'_gram']=matrix.tolist()
            for i,j in itertools.combinations(range(len(labels)),2):
                entry['pairs'].append({'quantity':quantity,'left':labels[i],'right':labels[j],**pair_from_gram(matrix,i,j)})
        results.append(entry);write('results.json',results)
        del master,bf16,vectors
    write('run_complete.json',{'status':'complete','elapsed_seconds':time.time()-started,
        'banks':2,'results_sha256':digest(OUT/'results.json'),'source_sha256':digest(ROOT/'direct_update_comparison.py'),
        'scope':'Direct non-fused virtual Adam comparisons; same saved state, weights, prefixes, and original paired PG actions',
        'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated()})
    status('complete')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():status('failed',error=repr(exc))
        raise
