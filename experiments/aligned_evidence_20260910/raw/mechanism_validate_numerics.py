"""Manually invoked numerical preflight for this frozen local experiment."""
import copy
import json
import math

import numpy as np
import torch

from measure import ROOT, adam_chunk, describe, gram, overlap, local_distillation_loss


torch.set_num_threads(2)
torch.manual_seed(947)
checks=[]
for mode in ['history','fresh','reset_m']:
    for use_zero in [False,True]:
        x=torch.nn.Parameter(torch.randn(137))
        optimizer=torch.optim.AdamW([x],lr=.003,betas=(.9,.98),eps=1e-8,weight_decay=.02,foreach=False,fused=False)
        for _ in range(3):
            x.grad=torch.randn_like(x);optimizer.step()
        state=optimizer.state[x]
        if mode=='fresh':
            state['exp_avg'].zero_();state['exp_avg_sq'].zero_();state['step'].zero_()
        elif mode=='reset_m':state['exp_avg'].zero_()
        exported={'value':x.detach().clone(),'exp_avg':state['exp_avg'].clone(),'exp_avg_sq':state['exp_avg_sq'].clone(),
                  'step':int(state['step']),'betas':(.9,.98),'lr':.003,'eps':1e-8,'weight_decay':.02,'bias_correction':True}
        gradient=torch.zeros_like(x) if use_zero else torch.randn_like(x)
        predicted=adam_chunk(exported,gradient,'history')
        x.grad=gradient;optimizer.step()
        torch.testing.assert_close(predicted,x,rtol=2e-6,atol=3e-7)
        torch.testing.assert_close(predicted.bfloat16(),x.bfloat16(),rtol=0,atol=0)
        checks.append({'check':'torch_adamw_reference','mode':mode,'zero_gradient':use_zero,
                       'max_error':float((predicted-x).abs().max().detach())})
# Check actual reset branches against explicitly modified states.
state={'value':torch.randn(29),'exp_avg':torch.randn(29),'exp_avg_sq':torch.rand(29),
       'step':10,'betas':(.9,.98),'lr':.003,'eps':1e-8,'weight_decay':.02}
g=torch.randn(29)
for mode in ['fresh','reset_m']:
    manual=copy.deepcopy(state);manual['exp_avg'].zero_()
    if mode=='fresh':manual['exp_avg_sq'].zero_();manual['step']=0
    torch.testing.assert_close(adam_chunk(state,g,mode),adam_chunk(manual,g),rtol=0,atol=0)
    checks.append({'check':'reset_branch','mode':mode})

logits=torch.randn(3,9,requires_grad=True)
logq=torch.randn(3,9).log_softmax(-1)
w=torch.tensor([.2,.3,.5])
full=local_distillation_loss(logits,logq,loss='full_vocab',weights=w)
full_g=torch.autograd.grad(full,logits)[0]
expected=torch.zeros_like(logits)
for token in range(9):
    lp=logits.log_softmax(-1)
    advantage=(logq[:,token]-lp[:,token]).detach()
    loss=-(w*lp.detach().exp()[:,token]*advantage*lp[:,token]).sum()
    expected+=torch.autograd.grad(loss,logits)[0]
torch.testing.assert_close(expected,full_g,atol=2e-7,rtol=2e-6)
checks.append({'check':'exact_enumeration_pg_expectation','max_error':float((expected-full_g).abs().max())})
for loss in ['teacher_topk','topk_intersection']:
    k=4
    actual=local_distillation_loss(logits,logq,loss=loss,weights=w,topk=k)
    actual_g=torch.autograd.grad(actual,logits)[0]
    lp=logits.log_softmax(-1)
    qv,qi=logq.topk(k,-1)
    if loss=='teacher_topk':
        pv=lp.gather(-1,qi)
        reference=(w[:,None]*(pv.exp()*(pv-qv)-pv.exp()+qv.exp())).sum()
    else:
        pv,pi=lp.topk(k,-1)
        mask=(pi[:,:,None]==qi[:,None,:]).any(-1)
        coeff=(pv.softmax(-1)*(logq.gather(-1,pi)-pv)*mask).detach()
        reference=-(w[:,None]*coeff*pv).sum()
    reference_g=torch.autograd.grad(reference,logits)[0]
    torch.testing.assert_close(actual_g,reference_g,rtol=2e-6,atol=2e-7)
    checks.append({'check':'independent_loss_formula','loss':loss})

vectors=[torch.randn(1031),torch.randn(1031),torch.zeros(1031)]
observed=gram(vectors,'cpu')
reference=np.stack([v.numpy().astype('float64') for v in vectors])
np.testing.assert_allclose(observed,reference@reference.T,rtol=1e-12,atol=1e-12)
checks.append({'check':'full_coordinate_gram'})
for vector in [vectors[0],torch.ones(1031),vectors[2]]:
    met,packed=describe(vector,True)
    for fraction in [.001,.01]:
        count=max(1,math.ceil(len(vector)*fraction)) if vector.any() else 0
        ids=np.argsort(-np.abs(vector.numpy()),kind='stable')[:count]
        expected=np.zeros(len(vector),dtype=bool);expected[ids]=True
        np.testing.assert_array_equal(np.unpackbits(packed[str(fraction)])[:len(vector)],expected)
        got=overlap(packed[str(fraction)],packed[str(fraction)],count,count,len(vector))
        assert got['jaccard']==(1. if count else None)
        if vector.any():
            energy=float(vector.double().square().sum())
            fraction_energy=float(vector[ids].double().square().sum())/energy
            assert math.isclose(met['top_energy'][str(fraction)],fraction_energy,rel_tol=1e-6)
checks.append({'check':'support_ties_zero_vectors_energy_and_overlap'})
(ROOT/'numerical_validation.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2)+'\n')
print(json.dumps({'status':'passed','checks':len(checks)}))
