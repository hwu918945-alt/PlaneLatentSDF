"""Observed-plane figures and uncapped meshes, plus separate heldout evaluation."""
import json
from pathlib import Path
import numpy as np
import torch
from .inference import restore,bounds,plane_grid,query
from .geometry import contour_labels,contour_report
from .extract_mesh import mesh_arrays


def labels(c,index,q,basis):
    if c.get('schema')=='plane_auto_occ_solid_v1':
        from .solid import SolidClassifier
        classifier=SolidClassifier(c['solid_rings_mm'][index])
        return classifier.classify(q[:,:2]*c['scale_mm']+c['center_mm'][:2])[0]==0
    return contour_labels(q,c['boundary'][index].numpy(),c['boundary_endo'][index].numpy(),basis)[0]==0


def dice(pred,truth): return float((2*(pred & truth).sum()+1e-6)/(pred.sum()+truth.sum()+1e-6))


def evaluate(checkpoint,case,output,device='cpu',resolution=160,mesh_resolution=96,
             observed_only=False,heldout_contours=None,domain=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    model,code,u,obs,full=restore(checkpoint,case,device)
    lower,upper,basis=bounds(obs,20.)
    if domain:
        # Frozen benchmark XY domain; query only inside the observed axial range.
        d=json.loads(Path(domain).read_text()); scale=obs['scale_mm']; center=obs['center_mm']
        corners=np.array([[x,y,z] for x in (d['lower_mm'][0],d['upper_mm'][0]) for y in (d['lower_mm'][1],d['upper_mm'][1])
            for z in (center[2],)])
        norm=(corners-center)/scale@basis
        lower[:2]=norm[:,:2].min(0); upper[:2]=norm[:,:2].max(0)
    rows=[]
    def analyze(pos,index=None,heldout=None):
        q=plane_grid(pos,lower,upper,basis,resolution); logits=query(model,q,obs,u); image=logits.reshape(resolution,resolution)
        pred=logits>=0; row=dict(position_mm=float(pos*obs['scale_mm']+obs['center_mm']@basis[:,2]),
            cached_plane_index=index,**contour_report(image))
        if index is not None:
            truth=labels(full,index,q,basis); row['kind']='observed' if index in obs['observed'] else 'strict_heldout'
        elif heldout:
            from .solid import SolidClassifier
            truth=SolidClassifier(heldout['rings_uv_mm']).classify(q[:,:2]*obs['scale_mm']+obs['center_mm'][:2])[0]==0
            row['kind']='strict_heldout'
        else: truth=None; row['kind']='unlabelled_gap'
        if truth is not None: row['dice']=dice(pred,truth)
        fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(image.T,origin='lower',cmap='RdBu',vmin=-6,vmax=6)
        if image.min()<0<image.max(): ax.contour(image.T,[0],colors=['cyan'],linewidths=.8)
        if truth is not None and truth.any() and not truth.all(): ax.contour(truth.reshape(image.shape).T,[.5],colors=['orange'],linewidths=.8)
        ax.set_title(f"{row['kind']} {row['position_mm']:.2f} mm"); ax.set_aspect('equal');fig.tight_layout()
        # All observed slices/gaps at milestones; limit heldout figure count only.
        if row['kind']!='strict_heldout' or len(rows)%10==0: fig.savefig(out/f'plane_{len(rows):03d}.png',dpi=120)
        plt.close(fig); rows.append(row)
    for i,pos in enumerate(full['s'].numpy()):
        if (not observed_only or i in obs['observed']) and lower[2]-1e-6<=pos<=upper[2]+1e-6: analyze(float(pos),i)
    for left,right in zip(obs['s'][:-1].cpu().numpy(),obs['s'][1:].cpu().numpy()): analyze(float((left+right)/2))
    if heldout_contours:
        if observed_only: raise ValueError('Milestones may not read heldout contours')
        planes=json.loads(Path(heldout_contours).read_text())['planes']
        for plane in planes:
            pos=(plane['height_mm']-obs['center_mm'][2])/obs['scale_mm']
            if not lower[2]-1e-6<=pos<=upper[2]+1e-6: raise ValueError('Heldout plane outside observed extent')
            analyze(float(pos),heldout=plane)
    # Native slab only, no mesh padding, smoothing, caps or repair.
    n=mesh_resolution; grid=[np.linspace(lower[i],upper[i],n) for i in range(3)]
    xx,yy=np.meshgrid(grid[0],grid[1],indexing='ij'); volume=np.empty((n,n,n),np.float32)
    for i,pos in enumerate(grid[2]):
        q=np.stack([xx.ravel(),yy.ravel(),np.full(xx.size,pos)],-1)@basis.T
        volume[:,:,i]=query(model,q,obs,u).reshape(n,n)
    report=dict(case_id=obs['case_id'],phase=obs['phase'],observed_indices=obs['observed'],planes=rows,
        threshold_logit=0,extent='observed axial range only',no_padding=True,no_artificial_caps=True)
    try:
        v,f,mreport=mesh_arrays(volume,lower,upper,basis,obs['center_mm'],obs['scale_mm'])
        with (out/'surface.obj').open('w') as file:
            file.write('# positive-inside occupancy, world benchmark mm; uncapped native slab\n')
            for vertex in v: file.write('v '+' '.join(f'{x:.8f}' for x in vertex)+'\n')
            for face in f+1: file.write('f '+' '.join(str(x) for x in face)+'\n')
        report['mesh']=mreport
        import trimesh
        mesh=trimesh.Trimesh(v,f,process=False)
        fig=plt.figure(figsize=(6,6)); ax=fig.add_subplot(111,projection='3d')
        count=min(len(f),12000); chosen=np.linspace(0,len(f)-1,count,dtype=int)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax.add_collection3d(Poly3DCollection(v[f[chosen]],facecolor='#58a7d6',edgecolor='none',alpha=.85))
        lo=v.min(0);hi=v.max(0);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(hi-lo)
        ax.view_init(elev=18,azim=35);fig.tight_layout();fig.savefig(out/'surface.png',dpi=140);plt.close(fig)
    except ValueError as error:
        report['mesh']=dict(status='NO_ZERO_CROSSING',reason=str(error))
    for kind in ('observed','strict_heldout'):
        values=[r['dice'] for r in rows if r['kind']==kind and 'dice' in r]
        report[kind+'_mean_dice']=float(np.mean(values)) if values else None
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    return report
