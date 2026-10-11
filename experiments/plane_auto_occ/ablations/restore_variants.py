from pathlib import Path
import torch
from ..inference import restore
from ..data import load_case
from ..area_pool import digest
from .model_case_only import CaseOnlyAutoOccupancy


def restore_variant(checkpoint,case,method='linear',device='cpu'):
    if method in ('linear','nearest'):
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ck.get('model_variant','plane_linear')=='case_only':raise ValueError('A checkpoint cannot impersonate C')
        return restore(checkpoint,case,device)
    if method!='case_only':raise ValueError('Unknown model variant')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['format']!='plane_auto_occ_case_only_v1' or ck['model_variant']!='case_only':raise ValueError('Not a case-only checkpoint')
    full=load_case(case);indices=[i for i,r in enumerate(ck['records']) if (r['case_id'],r['phase'])==(full['case_id'],full['phase'])]
    if len(indices)!=1:raise ValueError('Case not fitted in A checkpoint')
    index=indices[0];row=ck['records'][index]
    if digest(case)!=row['source_sha256']:raise ValueError('Source cache changed')
    obs=load_case(case,device,row['observed']);model=CaseOnlyAutoOccupancy(ck['config']['latent_dim'],ck['config']['width']).to(device).eval()
    model.load_state_dict(ck['model']);model.requires_grad_(False);code=ck['codes'][index].to(device)
    with torch.no_grad():u=model.plane_latents(code,obs['s'],obs['phase'])
    return model,code,u,obs,full
