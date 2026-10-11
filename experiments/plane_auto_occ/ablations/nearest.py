import numpy as np
import torch
from ..inference import query


def nearest_plane_latent(q,s,u,axis):
    if len(s)<2 or not bool(torch.all(s[1:]>s[:-1])):raise ValueError('Need >=2 increasing observed heights')
    h=q@axis
    if bool(((h.detach()<s[0]-1e-5)|(h.detach()>s[-1]+1e-5)).any()):raise ValueError('No axial extrapolation')
    r=torch.searchsorted(s.contiguous(),h.detach().contiguous(),right=False).clamp(0,len(s)-1)
    left=(r-1).clamp(0,len(s)-1)
    # Compare against the midpoint rather than subtracting two rounded float32
    # distances. Exactly represented computed midpoint must choose the lower plane.
    midpoint=(s[left]+s[r])/2
    index=torch.where(h<=midpoint,left,r)
    return u[index]


@torch.no_grad()
def query_variant(model,q,obs,u,method='linear',chunk=32768):
    if method in ('linear','case_only'):return query(model,q,obs,u,chunk)
    if method!='nearest':raise ValueError('Unknown inference method')
    return torch.cat([model.decoder(torch.as_tensor(p,dtype=torch.float32,device=obs['s'].device).to(u.dtype),
        nearest_plane_latent(torch.as_tensor(p,dtype=torch.float32,device=obs['s'].device).to(u.dtype),obs['s'],u,obs['axis']),obs['phase']).cpu()
        for p in np.array_split(q,max(1,int(np.ceil(len(q)/chunk))))]).numpy()
