"""Uniform-plane contour-proxy Dice, closure and latent sensitivity audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .inference import restore,bounds,plane_grid,query
from .geometry import contour_labels,contour_report
from .model import loo


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--case',required=True)
    p.add_argument('--output',required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--resolution',type=int,default=160); p.add_argument('--margin-mm',type=float,default=5.)
    p.add_argument('--gap-samples',type=int,default=1); p.add_argument('--threads',type=int,default=2)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    if a.resolution < 8 or a.margin_mm <= 0 or a.gap_samples < 0: raise ValueError('Invalid grid options')
    model,code,u,obs,full=restore(a.checkpoint,a.case,a.device)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use new evaluation output')
    out.mkdir(parents=True,exist_ok=True)
    lower,upper,basis=bounds(obs,a.margin_mm)
    observed=obs['observed']; ul=loo(obs['s'],u)
    rows=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def analyze(pos,index=None):
        q=plane_grid(pos,lower,upper,basis,a.resolution)
        logits=query(model,q,obs,u); image=logits.reshape(a.resolution,a.resolution)
        row=dict(position_mm=float(pos*obs['scale_mm']),cached_plane_index=index,**contour_report(image))
        base=(logits >= 0)
        variants={'mean_plane':u.mean(0,keepdim=True).expand_as(u),'reversed_plane':u.flip(0),'zero_plane':torch.zeros_like(u)}
        row['latent_sensitivity']={name:float(np.mean((query(model,q,obs,v) >= 0) != base)) for name,v in variants.items()}
        gt=None
        if index is not None:
            region,_,_=contour_labels(q,full['boundary'][index].numpy(),full['boundary_endo'][index].numpy(),basis)
            gt=region == 0
            row['kind']='observed' if index in observed else 'strict_heldout'
            row['contour_proxy_dice']=float((2*np.sum(base & gt)+1e-6)/(base.sum()+gt.sum()+1e-6))
            gt_image=gt.reshape(image.shape)
            row['ground_truth_touches_border']=bool(np.r_[gt_image[0],gt_image[-1],gt_image[:,0],gt_image[:,-1]].any())
            if index in observed and 0 < observed.index(index) < len(observed)-1:
                j=observed.index(index)-1
                with torch.no_grad():
                    qt=torch.tensor(q,dtype=torch.float32,device=obs['s'].device)
                    lp=model.decoder(qt,ul[j].expand(len(qt),-1),obs['phase']).cpu().numpy()
                pb=lp >= 0
                row['training_loo_proxy_dice']=float((2*np.sum(pb & gt)+1e-6)/(pb.sum()+gt.sum()+1e-6))
                row['training_loo_curves']=contour_report(lp.reshape(image.shape))
        else: row['kind']='unlabelled_gap'
        fig,ax=plt.subplots(figsize=(4,4))
        ax.imshow(base.reshape(image.shape).T,origin='lower',cmap='Blues',vmin=0,vmax=1)
        if gt is not None and gt.any() and not gt.all():
            ax.contour(gt.reshape(image.shape).T,levels=[0.5],colors=['orange'],linewidths=1)
        ax.set_title(f'{row["kind"]} s={row["position_mm"]:.2f} mm'); ax.set_aspect('equal')
        fig.tight_layout(); fig.savefig(out/f'plane_{len(rows):03d}.png',dpi=140); plt.close(fig)
        rows.append(row)
    for i,pos in enumerate(full['s'].numpy()):
        if lower[2]-1e-6 <= pos <= upper[2]+1e-6: analyze(float(pos),i)
    for left,right in zip(obs['s'][:-1].cpu().numpy(),obs['s'][1:].cpu().numpy()):
        for t in np.linspace(0,1,a.gap_samples+2)[1:-1]: analyze(float((1-t)*left+t*right))
    report=dict(case_id=obs['case_id'],phase=obs['phase'],observed_indices=observed,
                evaluation_target='filled cached Epi minus filled cached Endo polygons; not original raster masks',
                threshold_logit=0,extent='observed axial range only',planes=rows)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
