"""Evaluate fitted field on observed, LOO and strictly unseen section positions."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from skimage.draw import polygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from core.data import load_case,resample_contour
from core.model import PlaneField,leave_one_out_latent
from core.mesh import volume_to_mesh,write_obj
from core.losses import coordinate_gradient
from geometry_adapter import section_rings,all_boundary


def sample_curve(curve,count=1024):
    """Arclength samples without inventing closing edges on open predictions."""
    if np.linalg.norm(curve[0]-curve[-1])<1e-4:
        return resample_contour(curve[:-1],count)
    vectors=np.diff(curve,axis=0)
    lengths=np.linalg.norm(vectors,axis=1)
    keep=lengths>1e-10
    starts=curve[:-1][keep];vectors=vectors[keep];lengths=lengths[keep]
    cumulative=np.r_[0.,np.cumsum(lengths)]
    positions=np.linspace(0,cumulative[-1],count,endpoint=False)
    index=np.searchsorted(cumulative,positions,side='right')-1
    return starts[index]+((positions-cumulative[index])/lengths[index])[:,None]*vectors[index]


def mask_rings(rings,lower,spacing,shape):
    mask=np.zeros(shape,bool)
    for ring in rings:
        coords=(np.asarray(ring)-lower)/spacing
        i,j=polygon(coords[:,0],coords[:,1],shape=shape)
        part=np.zeros(shape,bool);part[i,j]=True;mask^=part
    return mask


def query(model,case,z,world,latent=None):
    values=[]
    with torch.no_grad():
        for start in range(0,len(world),16384):
            q=torch.as_tensor((world[start:start+16384]-case['center_mm'])/100,dtype=torch.float32,device=z.device)
            f=model.field(q,case['s'],z,case['axis'],case['phase']) if latent is None else model.decoder(q,latent.expand(len(q),-1),case['phase'])
            values.append(f.cpu().numpy())
    result=np.concatenate(values)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite inference')
    return result


def slice_metrics(model,case,z,plane,lower,upper,spacing=.75,latent=None):
    shape=np.ceil((upper-lower)/spacing).astype(int)+1
    axes=[lower[i]+np.arange(shape[i])*spacing for i in range(2)]
    xy=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
    world=np.c_[xy.reshape(-1,2),np.full(np.prod(shape),plane['height_mm'])]
    field=query(model,case,z,world,latent).reshape(tuple(shape))
    true=[np.asarray(r,float) for r in plane['rings_uv_mm']]
    gt=mask_rings(true,lower,spacing,tuple(shape));prediction=field<0
    denom=gt.sum()+prediction.sum();dice=float(2*(gt&prediction).sum()/denom) if denom else 1.
    curves=find_contours(field,0) if field.min()<0<field.max() else []
    predicted=[lower+curve*spacing for curve in curves]
    closed=all(np.linalg.norm(c[0]-c[-1])<1e-4 for c in predicted)
    mcd=hd95=None
    if predicted:
        gt_points=all_boundary(true,2048)
        sampled=[]
        for c in predicted:
            if len(c)>=3:sampled.append(sample_curve(c))
        if sampled:
            pred_points=np.concatenate(sampled)
            a=cKDTree(pred_points).query(gt_points)[0];b=cKDTree(gt_points).query(pred_points)[0]
            mcd=float((a.mean()+b.mean())/2);hd95=float(np.quantile(np.r_[a,b],.95))
    return dict(height_mm=plane['height_mm'],dice=dice,mcd_mm=mcd,hd95_mm=hd95,
                expected_boundary_loops=len(true),predicted_boundary_loops=len(curves),closed=bool(curves and closed),
                field_min=float(field.min()),field_max=float(field.max())),field,predicted,true


def mesh_distances(gt,pred):
    # True point-to-triangle distance in both directions, not nearest vertices.
    samples=5000
    a=trimesh.sample.sample_surface(gt,samples,seed=2301)[0]
    b=trimesh.sample.sample_surface(pred,samples,seed=2302)[0]
    distances=[]
    for points,target in [(a,pred),(b,gt)]:
        rows=[]
        for start in range(0,len(points),128):rows.append(trimesh.proximity.closest_point(target,points[start:start+128])[1])
        distances.append(np.concatenate(rows))
    return dict(mcd_mm=float((distances[0].mean()+distances[1].mean())/2),hd95_mm=float(np.quantile(np.concatenate(distances),.95)),
                samples_per_direction=samples,distance_definition='uniform-area surface samples to exact opposite mesh triangles, symmetric mean and pooled HD95')


def summarize(rows):
    metrics={}
    for key in ['dice','mcd_mm','hd95_mm']:
        vals=[r[key] for r in rows if r[key] is not None]
        metrics[key]=float(np.mean(vals)) if vals else None
    metrics.update(planes=len(rows),contour_coverage=sum(r['closed'] for r in rows)/len(rows),
                   boundary_loop_count_match=sum(r['expected_boundary_loops']==r['predicted_boundary_loops'] for r in rows)/len(rows))
    return metrics


def plot_mesh(ax,mesh,lower,upper,title):
    triangles=mesh.vertices[mesh.faces]
    normals=mesh.face_normals
    light=np.array([-.3,-.5,.8]);light/=np.linalg.norm(light)
    shade=.45+.55*np.clip(normals@light,0,1)
    color=np.array([.24,.62,.79])
    collection=Poly3DCollection(triangles,facecolors=shade[:,None]*color,edgecolors='none',rasterized=True)
    ax.add_collection3d(collection)
    for setter,a,b in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],lower,upper):setter(a,b)
    ax.set_box_aspect(upper-lower);ax.view_init(elev=18,azim=-70);ax.set_axis_off();ax.set_title(title)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='runs/main/last.pt')
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--domain',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);device='cuda' if torch.cuda.is_available() else 'cpu'
    saved=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=PlaneField(**saved['config']['model']).to(device).eval();model.load_state_dict(saved['student'])
    case=load_case('data/balloon_dog_observed.npz',device)
    with torch.no_grad():z=model.encoder(case['q'],case['sdf2d'],case['phase']);zl=leave_one_out_latent(case['s'],z)
    reference=trimesh.load('source/balloon_dog_reference_mm.ply',process=False)
    boundary=case['boundary'].cpu().numpy()*100+case['center_mm']
    lower=boundary.reshape(-1,3).min(0)-15;upper=boundary.reshape(-1,3).max(0)+15
    lower[2]=float(case['s'][0])*100+case['center_mm'][2];upper[2]=float(case['s'][-1])*100+case['center_mm'][2]
    domain=json.loads(Path(args.domain).read_text())
    lower=np.asarray(domain['lower_mm']);upper=np.asarray(domain['upper_mm'])
    observed=json.loads(Path('data/observed_contours.json').read_text())['planes']
    heldout=json.loads(Path('data/heldout_contours.json').read_text())['planes']
    groups={'observed':[],'loo':[],'heldout':[]};panels=[]
    for i,p in enumerate(observed):
        row,*_=slice_metrics(model,case,z,p,lower[:2],upper[:2]);groups['observed'].append(row)
        if 0<i<len(observed)-1:groups['loo'].append(slice_metrics(model,case,z,p,lower[:2],upper[:2],latent=zl[i-1:i])[0])
    for i,p in enumerate(heldout):
        result=slice_metrics(model,case,z,p,lower[:2],upper[:2]);groups['heldout'].append(result[0])
        if i in np.round(np.array([.05,.45,.72,.85,.92,.985])*(len(heldout)-1)).astype(int):panels.append(result)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'slice_metrics.json').write_text(json.dumps(groups,indent=2,allow_nan=False))
    print('Section evaluation',json.dumps({k:summarize(v) for k,v in groups.items()}),flush=True)
    resolution=160;shape=(resolution,)*3
    volume=np.empty(np.prod(shape),np.float32)
    for start in range(0,len(volume),16384):
        stop=min(start+16384,len(volume));index=np.stack(np.unravel_index(np.arange(start,stop),shape),-1)
        world=lower+index/(resolution-1)*(upper-lower);volume[start:stop]=query(model,case,z,world)
    vertices,faces,mesh_report=volume_to_mesh(volume.reshape(shape),(lower-case['center_mm'])/100,(upper-case['center_mm'])/100,np.eye(3),case['center_mm'],100.)
    mesh_report.update(resolution=resolution,artificial_axial_caps=False,coordinate_frame='declared benchmark millimetres',grid_spacing_mm=((upper-lower)/(resolution-1)).tolist())
    write_obj(out/'balloon_dog_reconstruction.obj',vertices,faces)
    pred=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    # Same uncapped triangle clipping, using the lower-level operation which
    # does not import Shapely's polygon/cap implementation.
    slab_vertices,slab_faces=reference.vertices.copy(),reference.faces.copy()
    for normal,origin in [([0,0,1],[0,0,lower[2]]),([0,0,-1],[0,0,upper[2]])]:
        slab_vertices,slab_faces,_=trimesh.intersections.slice_faces_plane(
            slab_vertices,slab_faces,np.asarray(normal),np.asarray(origin))
    slab=trimesh.Trimesh(vertices=slab_vertices,faces=slab_faces,process=False)
    slab.export(out/'balloon_dog_reference_observed_slab.obj')
    (out/'balloon_dog_reconstruction.report.json').write_text(json.dumps(mesh_report,indent=2))
    surface=mesh_distances(slab,pred)
    voxel_size=96;axes=[np.linspace(lower[i],upper[i],voxel_size) for i in range(3)]
    xy=np.stack(np.meshgrid(axes[0],axes[1],indexing='ij'),-1)
    spacing=(upper[:2]-lower[:2])/(voxel_size-1)
    intersection=gt_size=pred_size=0
    for height in axes[2]:
        rr=section_rings(reference,height)
        gt=mask_rings(rr,lower[:2],spacing,(voxel_size,voxel_size))
        world=np.c_[xy.reshape(-1,2),np.full(voxel_size**2,height)]
        prediction=query(model,case,z,world).reshape(voxel_size,voxel_size)<0
        intersection+=int((gt&prediction).sum());gt_size+=int(gt.sum());pred_size+=int(prediction.sum())
    voxel_dice=2*intersection/(gt_size+pred_size) if gt_size+pred_size else 1.
    q=(case['boundary'][:-1,::8]+case['boundary'][1:,::8])/2
    q=q.reshape(-1,3).detach().requires_grad_(True)
    grad=coordinate_gradient(model.field(q,case['s'],z.detach(),case['axis'],case['phase']),q,False).norm(dim=-1).detach().cpu().numpy()
    summary=dict(case='balloon_dog',optimization_steps=saved['global_step'],observed_planes=len(observed),heldout_planes=len(heldout),
                 evaluation_domain=domain,checkpoint=args.checkpoint,
                 sections={k:summarize(v) for k,v in groups.items()},voxel_dice_observed_slab=voxel_dice,voxel_resolution=voxel_size,
                 surface_observed_slab=surface,mesh=mesh_report,gap_gradient_norm_quantiles=np.quantile(grad,[0,.05,.5,.95,1]).tolist(),
                 gt_reference='OReX Balloon Dog reference; no hole repair; surface evaluation restricted to observed axial slab',
                 heldout_used_for_training_or_selection=False,full_reference_mesh_used_for_loss=False,
                 section_distance_definition='2048 perimeter-proportional reference samples and 1024 arclength samples per predicted curve; symmetric sampled-point MCD and pooled HD95; missing contours excluded from distance averages and recorded by coverage',
                 mesh_connected_components=int(len(pred.split(only_watertight=False))))
    (out/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    fig=plt.figure(figsize=(15,6))
    plot_mesh(fig.add_subplot(131,projection='3d'),slab,lower,upper,'Reference: observed slab')
    ax=fig.add_subplot(132,projection='3d')
    for p in observed:
        for ring in p['rings_uv_mm']:
            r=np.asarray(ring);r=np.r_[r,r[:1]];ax.plot(r[:,0],r[:,1],np.full(len(r),p['height_mm']),linewidth=1,color='#d75272')
    for setter,a,b in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],lower,upper):setter(a,b)
    ax.set_box_aspect(upper-lower);ax.view_init(elev=18,azim=-70);ax.set_axis_off();ax.set_title(f'{len(observed)} observed planes')
    plot_mesh(fig.add_subplot(133,projection='3d'),pred,lower,upper,'Single-case reconstruction')
    fig.tight_layout();fig.savefig(out/'balloon_dog_comparison.png',dpi=180,transparent=True);plt.close(fig)
    fig,axes_plot=plt.subplots(2,3,figsize=(12,9))
    for ax,(row,field,pred_rings,true) in zip(axes_plot.flat,panels):
        for ring in true:r=np.r_[ring,ring[:1]];ax.plot(r[:,0],r[:,1],'k-',linewidth=1)
        for ring in pred_rings:ax.plot(ring[:,0],ring[:,1],color='tab:green',linewidth=1)
        ax.set_xlim(lower[0],upper[0]);ax.set_ylim(lower[1],upper[1]);ax.set_aspect('equal')
        ax.set_title(f'Unseen z={row["height_mm"]:.1f}; Dice={row["dice"]:.3f}')
    fig.suptitle('Black: reference section; green: fitted zero contour');fig.tight_layout();fig.savefig(out/'heldout_contours.png',dpi=150);plt.close(fig)
    (out/'EVALUATION_COMPLETE.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
