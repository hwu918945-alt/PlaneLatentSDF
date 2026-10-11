"""Read-only milestone audit; only observed contours and positions are accessed.

Run in a subprocess so sampling/model construction do not alter training RNG.
All grids, interpolation nodes, and supervision use the checkpoint observed list.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from skimage.measure import euler_number
from .data import load_case, sample_plane
from .model import AutoOccupancy, loo
from .inference import bounds, plane_grid, query
from .geometry import contour_labels, contour_report
from .extract_mesh import mesh_arrays


def topology(image):
    report=contour_report(image)
    report['positive_euler_number']=int(euler_number(image>0,connectivity=1))
    report['single_connected_ring']=bool(report['positive_components']==1 and
        report['positive_euler_number']==0 and report['closed_curves']==2 and
        report['open_curves']==0 and not report['positive_touches_border'])
    return report


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','case','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu'); p.add_argument('--resolution',type=int,default=96)
    a=p.parse_args(); torch.set_num_threads(2); torch.manual_seed(1911)
    if a.resolution<8: raise ValueError('Invalid grid')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck['format']!='plane_auto_occ_v1' or len(ck['records'])!=1: raise ValueError('Single-case checkpoint required')
    import hashlib
    if hashlib.sha256(Path(a.case).read_bytes()).hexdigest()!=ck['records'][0]['source_sha256']:
        raise ValueError('Cache fingerprint changed')
    c=load_case(a.case,a.device,ck['records'][0]['observed'])
    if 'sampling_pool' in ck:
        from .area_pool import AreaSampler,digest
        from .paths import resolve_sampling_pool
        pool=ck['sampling_pool']
        pool_path=resolve_sampling_pool(a.checkpoint,pool)
        if digest(pool_path)!=pool['sha256']: raise ValueError('Diagnostic pool hash changed')
        c['_area_sampler']=AreaSampler(pool_path,1911)
    cfg=ck['config']; model=AutoOccupancy(cfg['latent_dim'],cfg['width']).to(a.device).eval()
    model.load_state_dict(ck['model']); code=ck['codes'][0].to(a.device).requires_grad_()
    u=model.plane_latents(code,c['s'],c['phase']); ul=loo(c['s'],u)
    groups={'code':[code],'generator':list(model.generator.parameters()),'decoder':list(model.decoder.parameters())}
    params=sum(groups.values(),[]); direct=[]; leaveout=[]; fallbacks=0
    for i in range(len(c['s'])):
        q,y,f=sample_plane(c,i,min(cfg['points'],4096)); fallbacks+=f
        direct.append(F.binary_cross_entropy_with_logits(model.decoder(q,u[i].expand(len(q),-1),c['phase']),y))
        if 0<i<len(c['s'])-1:
            leaveout.append(F.binary_cross_entropy_with_logits(model.decoder(q,ul[i-1].expand(len(q),-1),c['phase']),y))
    losses={'direct':torch.stack(direct).mean(),'loo':torch.stack(leaveout).mean() if leaveout else u.sum()*0}
    grads={}
    for name,loss in losses.items():
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        offset=0; grads[name]={}
        for group,ps in groups.items():
            selected=gs[offset:offset+len(ps)]; offset+=len(ps)
            grads[name][group]=float(torch.sqrt(sum((g.square().sum() for g in selected if g is not None),torch.tensor(0.,device=a.device))))
    report=dict(step=ck['step'],observed_indices=c['observed'],hidden_labels_used=False,
                diagnostic_points_per_plane=min(cfg['points'],4096),sampling_pool=ck.get('sampling_pool'),
                sampled_bce={k:float(v.detach()) for k,v in losses.items()},
                unweighted_gradient_norm=grads,pool_fallbacks=fallbacks,
                adjacent_latent_distances=(u[1:]-u[:-1]).norm(dim=-1).detach().cpu().tolist(),
                code_norm=float(code.detach().norm()),planes=[],gaps=[])
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    lower,upper,basis=bounds(c)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    u=u.detach(); ul=ul.detach()
    for i,pos in enumerate(c['s'].detach().cpu().numpy()):
        q=plane_grid(float(pos),lower,upper,basis,a.resolution)
        logits=query(model,q,c,u); image=logits.reshape(a.resolution,a.resolution)
        r,_,_=contour_labels(q,c['boundary'][i].detach().cpu().numpy(),c['boundary_endo'][i].detach().cpu().numpy(),basis)
        gt=r==0; pred=logits>=0
        row=dict(cached_plane_index=c['observed'][i],position_mm=float(pos*c['scale_mm']),
                 logit_quantiles=np.quantile(logits,[0,.01,.5,.99,1]).tolist(),
                 direct_dice=float((2*(pred & gt).sum()+1e-6)/(pred.sum()+gt.sum()+1e-6)),
                 direct_topology=topology(image),latent_sensitivity={})
        for name,v in {'mean_plane':u.mean(0,keepdim=True).expand_as(u),'reversed_plane':u.flip(0),'zero_plane':torch.zeros_like(u)}.items():
            lv=query(model,q,c,v)
            row['latent_sensitivity'][name]=dict(mean_absolute_logit_change=float(np.abs(lv-logits).mean()),
                                               classification_change=float(np.mean((lv>=0)!=pred)))
        with torch.no_grad():
            code_variants={'zero_case_code':torch.zeros_like(code)}
            for k in range(3): code_variants[f'random_case_code_{k}']=torch.randn_like(code)*.01
            for name,variant in code_variants.items():
                latent=model.plane_latents(variant,c['s'],c['phase'])
                lv=query(model,q,c,latent)
                row['latent_sensitivity'][name]=dict(mean_absolute_logit_change=float(np.abs(lv-logits).mean()),
                                                   classification_change=float(np.mean((lv>=0)!=pred)))
        row['other_case_code_available']=False
        images=[image]; labels=['Direct']
        if 0<i<len(c['s'])-1:
            with torch.no_grad():
                qt=torch.as_tensor(q,dtype=torch.float32,device=a.device)
                lp=model.decoder(qt,ul[i-1].expand(len(qt),-1),c['phase']).cpu().numpy()
            pb=lp>=0; row['training_loo_proxy_dice']=float((2*(pb & gt).sum()+1e-6)/(pb.sum()+gt.sum()+1e-6))
            row['loo_topology']=topology(lp.reshape(image.shape)); images.append(lp.reshape(image.shape)); labels.append('Training LOO')
        fig,axes=plt.subplots(1,len(images),figsize=(4*len(images),4),squeeze=False)
        for ax,img,label in zip(axes[0],images,labels):
            ax.imshow((img>=0).T,origin='lower',cmap='Blues',vmin=0,vmax=1)
            ax.contour(gt.reshape(image.shape).T,levels=[.5],colors=['orange'],linewidths=1)
            ax.set_title(f'{label} observed {c["observed"][i]}'); ax.set_aspect('equal')
        fig.tight_layout(); fig.savefig(out/f'observed_{c["observed"][i]:03d}.png',dpi=140); plt.close(fig)
        report['planes'].append(row)
    for left,right in zip(c['s'][:-1].detach().cpu().numpy(),c['s'][1:].detach().cpu().numpy()):
        pos=float((left+right)/2); q=plane_grid(pos,lower,upper,basis,a.resolution)
        image=query(model,q,c,u).reshape(a.resolution,a.resolution)
        report['gaps'].append(dict(position_mm=pos*c['scale_mm'],kind='unlabelled_gap',**topology(image)))
        fig,ax=plt.subplots(figsize=(4,4)); ax.imshow((image>=0).T,origin='lower',cmap='Blues',vmin=0,vmax=1)
        ax.set_title(f'Unlabelled gap {pos*c["scale_mm"]:.2f} mm'); fig.tight_layout()
        fig.savefig(out/f'gap_{len(report["gaps"]):03d}.png',dpi=140); plt.close(fig)
    n=40; axes=[np.linspace(lower[i],upper[i],n) for i in range(3)]
    xx,yy=np.meshgrid(axes[0],axes[1],indexing='ij'); volume=np.empty((n,n,n),np.float32)
    for i,pos in enumerate(axes[2]):
        q=np.stack([xx.ravel(),yy.ravel(),np.full(xx.size,pos)],-1) @ basis.T
        volume[:,:,i]=query(model,q,c,u).reshape(n,n)
    if volume.min()<0<volume.max():
        v,f,m=mesh_arrays(volume,lower,upper,basis,c['center_mm'],c['scale_mm'])
        with (out/'myocardium.obj').open('w') as file:
            for pt in v: file.write('v '+' '.join(str(float(x)) for x in pt)+'\n')
            for face in f+1: file.write('f '+' '.join(str(int(x)) for x in face)+'\n')
        report['mesh']=dict(status='extracted',resolution=n,no_padding=True,no_caps=True,**m)
    else:
        report['mesh']=dict(status='no_zero_crossing',field_min=float(volume.min()),field_max=float(volume.max()),resolution=n)
    (out/'diagnostics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(step=ck['step'],sampled_bce=report['sampled_bce'],mesh=report['mesh'])),flush=True)


if __name__=='__main__': main()
