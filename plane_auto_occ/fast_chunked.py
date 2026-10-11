"""Complete-case gradients with one generator graph and batched decoder queries.

Class-balanced BCE weights equal a mean over the unchanged 1/2,1/4,1/4
point budget. Detached plane latents accumulate decoder gradients before a
single chain-rule backward through G. No AMP, changed loss, or extra Adam step.
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model import loo


def backward_case(model,code,c,cfg,step):
    if cfg['seam_weight']!=0:raise ValueError('Fast backend requires Seam=0')
    points=cfg['points'];n=len(c['s']);device=code.device
    beta=float(cfg['loo_mix']) if cfg['loo_weight'] else 0.
    if points%4 or not 0<=beta<=1 or n<2 or (beta and n<3):raise ValueError('Invalid point/mix contract')
    sampling=[];qs=[];ys=[];start=time.perf_counter()
    for i in range(n):
        q,y=c['_area_sampler'].draw(i,points);qs.append(q);ys.append(y)
        sampling.append(c['_area_sampler'].last_stats[i])
    packed=torch.from_numpy(np.concatenate(qs));labels=torch.from_numpy(np.concatenate(ys))
    if device.type=='cuda':packed=packed.pin_memory();labels=labels.pin_memory()
    q=packed.to(device,non_blocking=True);y=labels.to(device,non_blocking=True)
    sampling_seconds=time.perf_counter()-start
    u=model.plane_latents(code,c['s'],c['phase']);leaf=u.detach().requires_grad_(True)
    direct_count=n*points;loo_count=(n-2)*points if beta else 0;count=direct_count+loo_count
    values=torch.empty(count,device=device);predictions=torch.empty(direct_count,device=device,dtype=torch.bool)
    total=torch.zeros((),device=device);batch=int(cfg.get('fast_query_points',65536))
    for offset in range(0,count,batch):
        index=torch.arange(offset,min(offset+batch,count),device=device)
        is_loo=index>=direct_count
        query=torch.where(is_loo,index-direct_count+points,index)
        plane=query//points
        conditions=torch.cat((leaf,loo(c['s'],leaf)),0) if beta else leaf
        condition=torch.where(is_loo,n+plane-1,plane)
        logits=model.decoder(q[query],conditions[condition],c['phase'])
        bce=F.binary_cross_entropy_with_logits(logits,y[query],reduction='none')
        weights=torch.where(is_loo,beta,torch.where((plane>0)&(plane<n-1),1-beta,1.)) if beta else torch.ones_like(bce)
        weighted=(bce*weights).sum()*(float(cfg['data_weight'])/(n*points))
        weighted.backward();total+=weighted.detach();values[offset:offset+len(index)]=bce.detach()
        # Direct queries precede LOO queries: slicing uses host offsets, no GPU scalar sync.
        direct_here=max(0,min(offset+len(index),direct_count)-offset)
        if direct_here:predictions[offset:offset+direct_here]=logits[:direct_here].detach()>=0
    u.backward(leaf.grad)
    zloss=code.square().mean();regularizer=cfg['latent_weight']*zloss;regularizer.backward()
    loss=total+regularizer.detach()
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite complete-case objective')
    # One bulk transfer for detailed metrics; no tensor-to-Python sync in query loop.
    v=values.cpu().numpy();pred=predictions.cpu().numpy().reshape(n,points)
    truth=np.concatenate(ys).reshape(n,points).astype(bool)
    direct=v[:direct_count].reshape(n,points).mean(1).tolist()
    leaves=v[direct_count:].reshape(n-2,points).mean(1).tolist() if beta else []
    dices=(2*(pred&truth).sum(1)+1e-6)/(pred.sum(1)+truth.sum(1)+1e-6)
    cuts=[slice(0,points//2),slice(points//2,3*points//4),slice(3*points//4,points)]
    names=c.get('region_names',['myocardium','cavity','background'])
    accuracies=[dict(index=c['observed'][i],**{name:float((pred[i,s]==truth[i,s]).mean()) for name,s in zip(names,cuts)}) for i in range(n)]
    stats=dict(direct=float(np.mean(direct)),loo=float(np.mean(leaves)) if leaves else 0.,data_loss=float(total),
        loo_mix=beta,seam=0.,latent=float(zloss.detach()),sampled_direct_dice=float(dices.mean()),
        adjacent_latent_distance=float((u.detach()[1:]-u.detach()[:-1]).norm(dim=-1).mean()),code_norm=float(code.detach().norm()),
        pool_fallbacks=0,plane_direct=direct,plane_loo=leaves,region_accuracy=accuracies,sampling=sampling,
        independent_points=points*n,decoder_queries=count,microbatch_points=batch,
        sampling_transfer_seconds=sampling_seconds,query_batches=(count+batch-1)//batch,
        objective_version=cfg['objective_version'])
    return loss,stats
