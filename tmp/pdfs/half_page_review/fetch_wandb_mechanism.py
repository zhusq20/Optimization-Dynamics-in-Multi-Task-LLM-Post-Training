from pathlib import Path
import json
import netrc
import requests
import concurrent.futures
import hashlib
from datetime import datetime,timezone

out=Path('output/half_page_mechanism/wandb')
runs=json.loads((out/'all_runs.json').read_text())
auth=('api',netrc.netrc().authenticators('api.wandb.ai')[2])
selected=['qyjllpm6','p5zo7hs6','iy0r84am','01ode9mp','2uqnli25','zcjbfocp']
def obj(x):return json.loads(x) if isinstance(x,str) else x
def fetch(rid):
    r=next(r for r in runs if r['name']==rid)
    histkeys=(obj(r['historyKeys']) or {}).get('keys',{})
    groups={'gradient':['_step','mopd/update','mopd/aggregate_grad_norm','mopd/aggregate_grad_clipped'],
            'domain':['_step']+[f'mopd/task/{d}/{m}' for d in ['math','code','if','science'] for m in ['token_share','mean_response_length','teacher_loss','prompt_share']],
            'clock':['_step','mopd/update']}
    # Per-key requests retain sparse checkpoint measurements without requiring
    # them to be logged on the same row as the training-update clock.
    for key in histkeys:
        if key.startswith('paper/sparsity/') and '/all/' in key and key.endswith(('/l2','/energy90_fraction','/sparsity_at_0')):
            groups[key]=['_step','paper/step',key]
    specs=[json.dumps({'keys':ks,'samples':10000}) for ks in groups.values()]
    q='query{project(name:"iclr2027-mopd-dynamics",entityName:"zsqzz"){run(name:'+json.dumps(rid)+'){sampledHistory(specs:'+json.dumps(specs)+')}}}'
    res=requests.post('https://api.wandb.ai/graphql',auth=auth,json={'query':q},timeout=60)
    res.raise_for_status();data=res.json()
    if data.get('errors'):raise RuntimeError(str(data['errors']))
    histories=data['data']['project']['run']['sampledHistory']
    config=obj(r['config'])
    fields=['mopd_loss','mopd_topk','mopd_reduction','optimizer','lr','mopd_output_dir','hf_checkpoint','seed','mopd_seed','clip_grad','adam_beta1','adam_beta2','rollout_max_response_len','mopd_total_steps']
    cfg={k:config.get(k,{}).get('value') for k in fields if isinstance(config.get(k),dict)}
    result={'run_id':rid,'name':r['displayName'],'state':r['state'],
        'url':f'https://wandb.ai/zsqzz/iclr2027-mopd-dynamics/runs/{rid}',
        'config':cfg,'retrieved_at':datetime.now(timezone.utc).isoformat(),
        'histories':dict(zip(groups,histories)),'expected_gradient_count':sum(x['count'] for x in histkeys['mopd/aggregate_grad_norm']['typeCounts'])}
    assert len(result['histories']['gradient'])==result['expected_gradient_count']
    f=out/(rid+'.json');f.write_text(json.dumps(result,indent=2))
    g=result['histories']['gradient'];d=result['histories']['domain']
    return {'run_id':rid,'name':r['displayName'],'gradient_count':len(g),
        'gradient_update_range':[min(x['mopd/update'] for x in g),max(x['mopd/update'] for x in g)],
        'domain_count':len(d),'sparse_geometry_series':len(groups)-3,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(fetch,selected))
(out/'retrieval_manifest.json').write_text(json.dumps(results,indent=2))
for r in results:print(json.dumps(r),flush=True)
