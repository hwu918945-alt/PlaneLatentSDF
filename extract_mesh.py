"""Extract p=0.5 (logit=0) myocardium surface; never pad/cap the slab."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from skimage.measure import marching_cubes
from .inference import restore,bounds,query


def mesh_arrays(volume,lower,upper,basis,center,scale):
    if not np.isfinite(volume).all() or not volume.min() < 0 < volume.max():
        raise ValueError('No finite logit zero crossing; cannot extract mesh')
    spacing=(upper-lower)/(np.array(volume.shape)-1)
    # Positive inside: ascent gives outward face winding in skimage's convention.
    # Verified explicitly on an analytic positive-inside sphere in tests.py.
    v,f,_,_=marching_cubes(volume,level=0,spacing=tuple(spacing),gradient_direction='ascent')
    stack_v=v+lower; world=(stack_v @ basis.T)*scale+center
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    contacts=(stack_v <= lower+spacing*.1)|(stack_v >= upper-spacing*.1)
    return world,f,dict(vertices=len(v),faces=len(f),boundary_edges=int((counts==1).sum()),
                        nonmanifold_edges=int((counts>2).sum()),watertight_edge_test=bool((counts==2).all()),
                        vertices_on_lateral_boundary=int(contacts[:,:2].any(1).sum()),
                        vertices_on_axial_boundary=int(contacts[:,2].sum()),
                        field_min=float(volume.min()),field_max=float(volume.max()))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--case',required=True)
    p.add_argument('--output',required=True); p.add_argument('--device',default='cpu'); p.add_argument('--resolution',type=int,default=128)
    p.add_argument('--margin-mm',type=float,default=5.); p.add_argument('--threads',type=int,default=2)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    if a.resolution < 8 or a.margin_mm <= 0: raise ValueError('Invalid mesh grid')
    model,code,u,obs,full=restore(a.checkpoint,a.case,a.device); lower,upper,basis=bounds(obs,a.margin_mm)
    out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    n=a.resolution; grid=[np.linspace(lower[i],upper[i],n) for i in range(3)]
    xx,yy=np.meshgrid(grid[0],grid[1],indexing='ij'); volume=np.empty((n,n,n),np.float32)
    for i,pos in enumerate(grid[2]):
        q=np.stack([xx.ravel(),yy.ravel(),np.full(xx.size,pos)],-1) @ basis.T
        volume[:,:,i]=query(model,q,obs,u).reshape(n,n)
    v,f,report=mesh_arrays(volume,lower,upper,basis,obs['center_mm'],obs['scale_mm'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as file:
        file.write('# aligned world millimetres; positive myocardium logit; no caps\n')
        for point in v: file.write('v '+' '.join(f'{x:.8f}' for x in point)+'\n')
        for face in f+1: file.write('f '+' '.join(str(x) for x in face)+'\n')
    report.update(no_padding=True,no_artificial_caps=True,axial_extent_mm=((upper[2]-lower[2])*obs['scale_mm']).item())
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
