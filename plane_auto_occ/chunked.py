"""Exactly one complete-case gradient, including code regularizer once."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from .model import loo


def backward_case(model,code,c,cfg,step):
    if cfg['seam_weight']!=0: raise ValueError('V2 chunked experiment requires Seam=0')
    points=cfg['points']; chunk=cfg['microbatch_points']; nplanes=len(c['s'])
    if chunk%4 or points%chunk: raise ValueError('Balanced chunks must divide plane budget')
    beta=float(cfg['loo_mix']) if cfg['loo_weight'] else 0.
    if not 0<=beta<=1 or nplanes<2 or (beta and nplanes<3): raise ValueError('Invalid mix/plane count')
    data_weight=float(cfg['data_weight']); device=c['axis'].device
    direct=[]; leaves=[]; dices=[]; region_accuracy=[]; total_value=0.; sampling=[]
    for i in range(nplanes):
        q,y=c['_area_sampler'].draw(i,points)
        sampling.append(c['_area_sampler'].last_stats[i])
        dvalue=0.; ovalue=0.; tp=0; positive=0; targets=0; correct=np.zeros(3,dtype=np.int64)
        budgets=[points//2,points//4,points//4]; cb=[chunk//2,chunk//4,chunk//4]
        for k in range(points//chunk):
            ix=np.concatenate([np.arange(sum(budgets[:r])+k*cb[r],sum(budgets[:r])+ (k+1)*cb[r]) for r in range(3)])
            qt=torch.as_tensor(q[ix],device=device); yt=torch.as_tensor(y[ix],device=device)
            # Recompute tiny generator graph; each backward frees its decoder graph.
            u=model.plane_latents(code,c['s'],c['phase'])
            logits=model.decoder(qt,u[i].expand(chunk,-1),c['phase'])
            spans=[slice(0,cb[0]),slice(cb[0],cb[0]+cb[1]),slice(cb[0]+cb[1],chunk)]
            d=sum(w*F.binary_cross_entropy_with_logits(logits[s],yt[s]) for w,s in zip((.5,.25,.25),spans))
            interior=0<i<nplanes-1 and beta>0
            if interior:
                latent=loo(c['s'],u)[i-1]
                lp=model.decoder(qt,latent.expand(chunk,-1),c['phase'])
                o=sum(w*F.binary_cross_entropy_with_logits(lp[s],yt[s]) for w,s in zip((.5,.25,.25),spans))
                contribution=(1-beta)*d+beta*o; ovalue+=float(o.detach())*chunk/points
            else: contribution=d
            weighted=contribution*(chunk/points)*(data_weight/nplanes)
            if not bool(torch.isfinite(weighted)): raise FloatingPointError(f'Nonfinite plane {i}, chunk {k}')
            weighted.backward(); total_value+=float(weighted.detach()); dvalue+=float(d.detach())*chunk/points
            pred=logits.detach()>=0; truth=yt.bool()
            tp+=int((pred & truth).sum());positive+=int(pred.sum());targets+=int(truth.sum())
            for r,s in enumerate(spans): correct[r]+=int((pred[s]==truth[s]).sum())
        direct.append(dvalue)
        if interior: leaves.append(ovalue)
        dices.append((2*tp+1e-6)/(positive+targets+1e-6))
        names=c.get('region_names',['myocardium','cavity','background'])
        region_accuracy.append(dict(index=c['observed'][i],**{name:float(correct[r]/budgets[r]) for r,name in enumerate(names)}))
    # Exactly once per optimizer update, with the unchanged mean-square units.
    zloss=code.square().mean(); regularizer=cfg['latent_weight']*zloss
    regularizer.backward(); loss_value=total_value+float(regularizer.detach())
    if not math.isfinite(loss_value): raise FloatingPointError('Nonfinite objective')
    with torch.no_grad(): u=model.plane_latents(code,c['s'],c['phase'])
    stats=dict(direct=float(np.mean(direct)),loo=float(np.mean(leaves)) if leaves else 0.,data_loss=total_value,
        loo_mix=beta,seam=0.,latent=float(zloss.detach()),sampled_direct_dice=float(np.mean(dices)),
        adjacent_latent_distance=float((u[1:]-u[:-1]).norm(dim=-1).mean()),code_norm=float(code.detach().norm()),
        pool_fallbacks=0,plane_direct=direct,plane_loo=leaves,region_accuracy=region_accuracy,sampling=sampling,
        independent_points=points*nplanes,decoder_queries=points*(nplanes+len(leaves)),microbatch_points=chunk,
        objective_version=cfg['objective_version'])
    return torch.tensor(loss_value,device=device),stats
