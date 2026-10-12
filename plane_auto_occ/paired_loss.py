"""Standard all-point BCE plus whole-plane Soft Dice on ROI-source points only."""
import math
import torch
from torch.nn import functional as F


ROI_SOURCE=4


def shape_loss(logits, targets, source, dice_weight=1.0, eps=1e-6):
    if logits.ndim!=1 or targets.shape!=logits.shape or source.shape!=logits.shape:
        raise ValueError('Expected one entire SAX plane, logits/targets/source [N]')
    if not math.isfinite(dice_weight) or dice_weight<0 or eps<=0:raise ValueError('Invalid Dice config')
    if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite logits')
    if not bool(((targets==0)|(targets==1)).all()):raise ValueError('Binary targets required')
    mask=source==ROI_SOURCE
    if not bool(mask.any()):raise ValueError('Missing independently tagged uniform ROI points')
    bce=F.binary_cross_entropy_with_logits(logits,targets)
    p=logits[mask].sigmoid();g=targets[mask]
    dice=1-(2*(p*g).mean()+eps)/(p.mean()+g.mean()+eps)
    return bce+dice_weight*dice,dict(bce=bce,dice_roi=dice)


def monitor(logits,targets,source,index,branch,std_threshold=1e-3):
    """Read-only, no early stop, adaptive loss, gate, or change to optimizer."""
    mask=source==ROI_SOURCE;p=logits.detach()[mask].sigmoid();g=targets[mask].bool();pred=p>=.5
    tp=(pred&g).sum();pos=pred.sum();gt=g.sum();std=p.std(unbiased=False)
    return dict(index=int(index),branch=branch,roi_probability_mean=float(p.mean()),roi_probability_std=float(std),
        roi_logit_std=float(logits.detach()[mask].std(unbiased=False)),
        roi_predicted_positive_fraction=float(pred.float().mean()),roi_gt_positive_fraction=float(g.float().mean()),
        roi_hard_dice=float((2*tp+1e-6)/(pos+gt+1e-6)),
        roi_precision=float(tp/pos) if pos else 0.,roi_recall=float(tp/gt) if gt else None,
        roi_pred_gt_area_ratio=float(pos/gt) if gt else None,
        empty_prediction=bool(pos==0),full_prediction=bool(pos==len(pred)),
        constant_candidate=bool(std<std_threshold),
        constant_threshold=float(std_threshold))


def _mix(cfg,n,i,branch):
    beta=float(cfg.get('loo_mix',.25)) if cfg['loo_weight'] else 0.
    if n<2 or (beta and n<3) or not 0<=beta<=1:raise ValueError('Invalid LOO contract')
    weight=beta if branch=='loo' else (1-beta if beta and 0<i<n-1 else 1.)
    return weight*float(cfg['data_weight'])/n


def backward_case(model,code,c,cfg,step):
    """Exact two-pass whole-plane Dice gradient with bounded decoder graphs.

    Pass 1 stores logits only; a small [N] leaf graph computes exact dL/dlogit.
    Pass 2 recomputes deterministic MLP chunks and applies those derivatives.
    One G backward, one latent-L2 contribution, then caller clips/steps once.
    Never calculate Dice independently per microbatch. No dropout/BN/AMP.
    """
    from .model import loo
    if cfg['seam_weight']!=0:raise ValueError('This exact backend requires existing Seam=0 protocol')
    device=code.device;points=int(cfg['points']);n=len(c['s']);batch=int(cfg.get('fast_query_points',65536))
    if points<1 or batch<1:raise ValueError('Positive point/query budgets required')
    beta=float(cfg['loo_mix']) if cfg['loo_weight'] else 0.
    u=model.plane_latents(code,c['s'],c['phase']);leaf=u.detach().requires_grad_(True)
    qs=[];ys=[];ss=[];sampling=[]
    for i in range(n):
        q,y=c['_area_sampler'].draw(i,points)
        source=c['_area_sampler'].last_source[i]
        qs.append(torch.as_tensor(q,device=device));ys.append(torch.as_tensor(y,device=device))
        ss.append(torch.as_tensor(source,device=device));sampling.append(c['_area_sampler'].last_stats[i])
    tasks=[(i,'direct') for i in range(n)]+([(i,'loo') for i in range(1,n-1)] if beta else [])
    direct=[];leaves=[];bces=[];dices=[];monitors=[];total=0.;queries=0
    dice_weight=float(cfg.get('dice_weight',0.))
    for i,branch in tasks:
        # Pass 1 is detached from model and generator, so memory is O(N logits).
        with torch.no_grad():
            latent=leaf[i] if branch=='direct' else loo(c['s'],leaf)[i-1]
            logits=torch.cat([model.decoder(qs[i][k:k+batch],latent.expand(min(batch,points-k),-1),c['phase'])
                              for k in range(0,points,batch)])
        proxy=logits.detach().requires_grad_(True)
        shape,parts=shape_loss(proxy,ys[i],ss[i],dice_weight,float(cfg.get('dice_eps',1e-6)))
        weight=_mix(cfg,n,i,branch)
        derivative=torch.autograd.grad(shape*weight,proxy)[0].detach()
        total+=float(shape.detach())*weight;value=float(shape.detach())
        (direct if branch=='direct' else leaves).append(value)
        bces.append(float(parts['bce'].detach()));dices.append(float(parts['dice_roi'].detach()))
        monitors.append(monitor(logits,ys[i],ss[i],c['observed'][i],branch,float(cfg.get('collapse_std_threshold',1e-3))))
        for k in range(0,points,batch):
            latent=leaf[i] if branch=='direct' else loo(c['s'],leaf)[i-1]
            a=model.decoder(qs[i][k:k+batch],latent.expand(min(batch,points-k),-1),c['phase'])
            a.backward(derivative[k:k+batch]);queries+=len(a)
    if leaf.grad is None:raise RuntimeError('No decoder-to-plane-latent gradient')
    u.backward(leaf.grad)
    zloss=code.square().mean();regularizer=cfg['latent_weight']*zloss;regularizer.backward()
    total+=float(regularizer.detach())
    if not math.isfinite(total):raise FloatingPointError('Nonfinite complete-case loss')
    stats=dict(direct=sum(direct)/len(direct),loo=sum(leaves)/len(leaves) if leaves else 0.,
        data_loss=total-float(regularizer.detach()),loo_mix=beta,seam=0.,latent=float(zloss.detach()),
        bce_mean=sum(bces)/len(bces),dice_roi_mean=sum(dices)/len(dices),dice_weight=dice_weight,
        plane_direct=direct,plane_loo=leaves,plane_monitor=monitors,
        sampled_direct_dice=sum(x['roi_hard_dice'] for x in monitors if x['branch']=='direct')/n,
        sampled_direct_dice_scope='ROI-uniform source only; not fixed-grid evaluation',
        adjacent_latent_distance=float((u.detach()[1:]-u.detach()[:-1]).norm(dim=-1).mean()),
        code_norm=float(code.detach().norm()),pool_fallbacks=0,sampling=sampling,
        independent_points=points*n,decoder_queries=queries,
        decoder_forward_evaluations=2*queries,microbatch_points=batch,
        objective_version=cfg['objective_version'],backward_version='whole_plane_exact_two_pass_v1')
    return torch.tensor(total,device=device,dtype=code.dtype),stats
