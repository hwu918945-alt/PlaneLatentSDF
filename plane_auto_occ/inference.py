import hashlib
from pathlib import Path
import numpy as np
import torch
from .data import load_case
from .model import AutoOccupancy
from .geometry import stack_basis


def restore(checkpoint, case_path, device='cpu'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['format'] != 'plane_auto_occ_v1': raise ValueError('Wrong checkpoint format')
    full=load_case(case_path)
    key=(full['case_id'],full['phase'])
    ids=[i for i,r in enumerate(ck['records']) if (r['case_id'],r['phase']) == key]
    if len(ids) != 1: raise ValueError('Case not fitted in checkpoint; run train --pretrained first')
    idx=ids[0]; row=ck['records'][idx]
    digest=hashlib.sha256(Path(case_path).read_bytes()).hexdigest()
    if digest != row['source_sha256']: raise ValueError('Case cache bytes differ from fit manifest')
    obs=load_case(case_path,device,row['observed'])
    model=AutoOccupancy(ck['config']['latent_dim'],ck['config']['width']).to(device).eval()
    model.load_state_dict(ck['model']); model.requires_grad_(False)
    code=ck['codes'][idx].to(device)
    with torch.no_grad(): u=model.plane_latents(code,obs['s'].float(),obs['phase'])
    return model,code,u,obs,full


def bounds(obs, margin_mm=5):
    basis=stack_basis(obs['axis'].detach().cpu().numpy())
    b=obs.get('geometry_points',obs['boundary']).detach().cpu().numpy().reshape(-1,3) @ basis
    lower=b.min(0); upper=b.max(0)
    lower[:2]-=margin_mm/obs['scale_mm']; upper[:2]+=margin_mm/obs['scale_mm']
    lower[2]=float(obs['s'][0]); upper[2]=float(obs['s'][-1])
    return lower,upper,basis


def plane_grid(pos, lower, upper, basis, resolution):
    x=np.linspace(lower[0],upper[0],resolution); y=np.linspace(lower[1],upper[1],resolution)
    xx,yy=np.meshgrid(x,y,indexing='ij')
    return np.stack([xx.ravel(),yy.ravel(),np.full(xx.size,pos)],-1) @ basis.T


@torch.no_grad()
def query(model,q,obs,u,chunk=32768):
    device=obs['s'].device
    return torch.cat([model.field(torch.as_tensor(p,dtype=torch.float32,device=device),obs['s'],u,
                                  obs['axis'],obs['phase']).cpu() for p in np.array_split(q,max(1,int(np.ceil(len(q)/chunk))))]).numpy()
