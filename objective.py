import torch
from torch.nn import functional as F
from .data import sample_plane
from .model import loo


def objective(model, code, c, cfg, step):
    u = model.plane_latents(code, c['s'].float(), c['phase'])
    ul = loo(c['s'], u)
    directs, leaveouts, seams = [], [], []
    dice, fallback = [], 0
    for i in range(len(c['s'])):
        q,y,f = sample_plane(c,i,cfg['points']); fallback += f
        logits = model.decoder(q,u[i].expand(len(q),-1),c['phase'])
        directs.append(F.binary_cross_entropy_with_logits(logits,y))
        pred=(logits.detach() >= 0).float()
        dice.append((2*(pred*y).sum()+1e-6)/(pred.sum()+y.sum()+1e-6))
        if 0 < i < len(c['s'])-1:
            lo=model.decoder(q,ul[i-1].expand(len(q),-1),c['phase'])
            leaveouts.append(F.binary_cross_entropy_with_logits(lo,y))
            if cfg['seam_weight'] > 0 and step >= cfg['seam_start']:
                eps=min(cfg['seam_eps_mm']/c['scale_mm'],
                        0.1*float(c['s'][i]-c['s'][i-1]),0.1*float(c['s'][i+1]-c['s'][i]))
                # BOTH sides must route through full piecewise interpolation.
                # No binary GT is attached to off-plane q±eps*axis.
                b=torch.cat((c['boundary'][i],c['boundary_endo'][i])).float()
                ids=torch.randint(len(b),(cfg['seam_points'],),device=b.device); b=b[ids]
                lp=model.field(b+eps*c['axis'],c['s'],u,c['axis'],c['phase'])
                lm=model.field(b-eps*c['axis'],c['s'],u,c['axis'],c['phase'])
                l0=model.field(b,c['s'],u,c['axis'],c['phase'])
                # Match slopes per millimetre; NOT a second derivative / eps².
                delta=(lp-2*l0+lm)/(eps*c['scale_mm'])
                seams.append(F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=1.))
    zero=u.sum()*0
    direct=torch.stack(directs).mean()
    looloss=torch.stack(leaveouts).mean() if leaveouts else zero
    seam=torch.stack(seams).mean() if seams else zero
    zloss=code.square().mean()
    total=cfg['direct_weight']*direct+cfg['loo_weight']*looloss+cfg['latent_weight']*zloss+cfg['seam_weight']*seam
    stats=dict(direct=float(direct.detach()),loo=float(looloss.detach()),seam=float(seam.detach()),
               latent=float(zloss.detach()),sampled_direct_dice=float(torch.stack(dice).mean()),
               adjacent_latent_distance=float((u[1:]-u[:-1]).norm(dim=-1).mean().detach()),
               code_norm=float(code.norm().detach()),pool_fallbacks=fallback)
    return total,stats
