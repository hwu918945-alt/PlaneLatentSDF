"""Matched ABC grids, native V2 queries/geometry/mesh, and explicit NN jumps."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import find_contours,euler_number
from ..inference import bounds,plane_grid
from ..geometry import contour_labels,contour_report,polygon_sdf
from ..extract_mesh import mesh_arrays
from ..area_pool import digest
from .restore_variants import restore_variant
from .nearest import query_variant
from .precision import promote

NAMES={'case_only':'A_case_only','nearest':'B_plane_nearest','linear':'C_plane_linear'}


def contour_metrics(logits,pos,index,full,obs,lower,upper,basis,resolution,region=None):
    image=logits.reshape(resolution,resolution);pred=logits>=0;report=contour_report(image)
    eu=int(euler_number(image>0,connectivity=1))
    report.update(index=index,cached_plane_index=index,position_mm=float(pos*obs['scale_mm']),
        kind='unlabelled_gap' if index is None else 'observed' if index in obs['observed'] else 'strict_heldout',
        positive_euler_number=eu,single_connected_ring=bool(report['curves']==2 and report['closed_curves']==2 and
        report['positive_components']==1 and eu==0 and not report['positive_touches_border']),
        logit_min=float(logits.min()),logit_max=float(logits.max()),contour_proxy_dice=None,
        precision=None,recall=None,predicted_to_gt_area_ratio=None,cavity_false_positive_rate=None,
        background_false_positive_rate=None,boundary_error=None)
    if index is None:return report,None
    if region is None:
        q=plane_grid(pos,lower,upper,basis,resolution)
        region,_,_=contour_labels(q,full['boundary'][index].numpy(),full['boundary_endo'][index].numpy(),basis)
    gt=region==0;tp=int((pred&gt).sum());npred=int(pred.sum());ngt=int(gt.sum())
    cellarea=float(np.prod((upper[:2]-lower[:2])*obs['scale_mm']/(resolution-1)))
    report.update(contour_proxy_dice=float((2*tp+1e-6)/(npred+ngt+1e-6)),precision=tp/npred if npred else None,
        recall=tp/ngt if ngt else None,predicted_to_gt_area_ratio=npred/ngt if ngt else None,
        predicted_area_mm2=npred*cellarea,gt_area_mm2=ngt*cellarea,
        cavity_false_positive_rate=float(pred[region==1].mean()) if np.any(region==1) else None,
        background_false_positive_rate=float(pred[region==2].mean()) if np.any(region==2) else None)
    probs=1/(1+np.exp(-np.clip(logits,-80,80)))
    report['region_statistics']={str(r):dict(count=int((region==r).sum()),mean_probability=float(probs[region==r].mean())) for r in (0,1,2)}
    curves=sorted(find_contours(image,0),key=lambda c:abs(np.sum(c[:,0]*np.roll(c[:,1],-1)-c[:,1]*np.roll(c[:,0],-1))),reverse=True)
    errors={}
    # Preserve failed components. Assign Epi/Endo only for a verified single ring.
    if report['single_connected_ring']:
        for name,curve,poly in zip(('epi','endo'),curves,(full['boundary'][index].numpy(),full['boundary_endo'][index].numpy())):
            xy=lower[:2]+curve/(resolution-1)*(upper[:2]-lower[:2])
            signed=polygon_sdf(xy,poly@basis[:,:2])*obs['scale_mm']
            errors[name]=dict(mean_signed_distance_pred_to_gt_mm=float(signed.mean()),
                mean_absolute_distance_pred_to_gt_mm=float(abs(signed).mean()),samples=len(curve),
                direction='prediction vertices -> matching filled GT polygon; positive outside, negative inside',
                matching='largest closed loop Epi, smaller Endo, only after verifying one connected ring',
                averaging='native contour vertices, one direction; NOT symmetric ASSD')
    else:
        errors={name:dict(mean_signed_distance_pred_to_gt_mm=None,mean_absolute_distance_pred_to_gt_mm=None,
            reason='No unambiguous single-ring Epi/Endo assignment; all components retained') for name in ('epi','endo')}
    report['boundary_error']=errors
    return report,gt.reshape(image.shape)


def mesh_report(volume,lower,upper,basis,obs,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if not volume.min()<0<volume.max():
        report=dict(status='NO_ZERO_CROSSING',field_min=float(volume.min()),field_max=float(volume.max()),
            no_padding=True,no_artificial_caps=True)
        target.with_suffix('.json').write_text(json.dumps(report,indent=2));return report,None
    v,f,report=mesh_arrays(volume,lower,upper,basis,obs['center_mm'],obs['scale_mm'])
    with target.open('w') as file:
        file.write('# native uncapped occupancy zero surface; aligned world millimetres\n')
        for vertex in v:file.write('v '+' '.join(f'{x:.8f}' for x in vertex)+'\n')
        for face in f+1:file.write('f '+' '.join(str(x) for x in face)+'\n')
    stack=(v-obs['center_mm'])/obs['scale_mm']@basis
    edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);edges,counts=np.unique(edges,axis=0,return_counts=True)
    adjacency=coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v)))
    nc,components=connected_components(adjacency,directed=False)
    be=edges[counts==1];n=volume.shape[0];eps=(upper-lower)/(n-1)*.1
    axial=np.all((abs(stack[be,2]-lower[2])<=eps[2]),1)|np.all((abs(stack[be,2]-upper[2])<=eps[2]),1)
    lateral=np.zeros(len(be),bool)
    for k in (0,1):lateral|=np.all(abs(stack[be,k]-lower[k])<=eps[k],1)|np.all(abs(stack[be,k]-upper[k])<=eps[k],1)
    if len(be):
        bv,inv=np.unique(be,return_inverse=True);bend=inv.reshape(-1,2)
        badj=coo_matrix((np.ones(2*len(be)),(np.r_[bend[:,0],bend[:,1]],np.r_[bend[:,1],bend[:,0]])),shape=(len(bv),len(bv)))
        nb,_=connected_components(badj,directed=False);degrees=np.bincount(inv.ravel(),minlength=len(bv))
    else:nb=0;degrees=np.array([],int)
    normals=np.cross(stack[f[:,1]]-stack[f[:,0]],stack[f[:,2]]-stack[f[:,0]])
    lengths=np.linalg.norm(normals,axis=1);normalz=np.divide(abs(normals[:,2]),lengths,out=np.zeros_like(lengths),where=lengths>0)
    dz=(upper[2]-lower[2])/(n-1);switches=[]
    for mid in ((obs['s'][:-1]+obs['s'][1:])/2).cpu().numpy():
        close=np.max(abs(stack[f,2]-mid),axis=1)<dz
        horizontal=normalz>.95
        switches.append(dict(position_mm=float(mid*obs['scale_mm']),faces_within_one_axial_voxel=int(close.sum()),
            near_horizontal_faces=int((close&horizontal).sum()),
            near_horizontal_area_mm2=float((lengths[close&horizontal]*obs['scale_mm']**2/2).sum())))
    report.update(status='EXTRACTED',no_padding=True,no_artificial_caps=True,axial_extent_mm=float((upper[2]-lower[2])*obs['scale_mm']),
        connected_components=int(nc),component_vertices=sorted(np.bincount(components).tolist(),reverse=True),
        boundary_edge_locations=dict(axial=int(axial.sum()),lateral=int((~axial&lateral).sum()),other=int((~axial&~lateral).sum())),
        boundary_components=int(nb),boundary_vertices_all_degree2=bool(np.all(degrees==2)),
        expected_four_axial_loops=bool(nb==4 and np.all(degrees==2) and axial.all()),switch_plane_face_statistics=switches)
    target.with_suffix('.json').write_text(json.dumps(report,indent=2));return report,(v,f)


def discontinuity(items,obs,lower,upper,basis,resolution):
    report=[];eps=.1/obs['scale_mm']
    for i,(left,right) in enumerate(zip(obs['s'][:-1],obs['s'][1:])):
        mid=float((left+right)/2);q=plane_grid(mid,lower,upper,basis,resolution);row=dict(position_mm=mid*obs['scale_mm'],offset_mm=.1,variants={})
        for method,(model,code,u,o,full) in items.items():
            center=query_variant(model,q,o,u,method)
            qm=q-eps*basis[:,2];qp=q+eps*basis[:,2]
            lm=query_variant(model,qm,o,u,method);lp=query_variant(model,qp,o,u,method)
            value=dict(spatial_mean_abs_logit_change=float(abs(lp-lm).mean()),spatial_classification_flip_rate=float(np.mean((lp>=0)!=(lm>=0))))
            if method in ('nearest','linear'):
                with torch.no_grad():
                    qt=torch.as_tensor(q,dtype=torch.float32,device=o['s'].device)
                    cl=model.decoder(qt,u[i].expand(len(q),-1),o['phase']).cpu().numpy()
                    cr=model.decoder(qt,u[i+1].expand(len(q),-1),o['phase']).cpu().numpy()
                value.update(fixed_xyz_condition_mean_abs_logit_jump=float(abs(cr-cl).mean()),
                    fixed_xyz_condition_flip_rate=float(np.mean((cr>=0)!=(cl>=0))))
                if method=='nearest':
                    value['midpoint_logit_max_abs_diff_from_left']=float(abs(center-cl).max())
                    heights=qt.to(o['s'].dtype)@o['axis'];nominal=(o['s'][i]+o['s'][i+1])/2
                    value['rounded_midpoint_plane_lower_assignments']=int((heights<=nominal).sum())
                    value['rounded_midpoint_plane_upper_assignments']=int((heights>nominal).sum())
                    value['rounded_midpoint_height_error_max_mm']=float(abs(heights-nominal).max()*o['scale_mm'])
                    value['rounding_note']='The nominal XY plane is rounded to native float32 xyz; some points fall above/below the exact midpoint. Exact computed ties always select lower.'
            row['variants'][NAMES[method]]=value
        report.append(row)
    return dict(tie_rule='left/lower observed plane at exact midpoint',queries='same XY and xyz; no hidden GT needed',planes=report)


def evaluate(case,c_checkpoint,output,a_checkpoint=None,device='cpu',resolution=160,mesh_resolution=128,
             margin_mm=5.,robust_margin_mm=20.,reference_metrics=None,reference_mesh=None,reference_audit=None,with_mesh=True):
    torch.set_num_threads(2);out=Path(output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('New evaluation directory required')
    out.mkdir(parents=True,exist_ok=True);sourcehash=digest(c_checkpoint)
    items={'nearest':restore_variant(c_checkpoint,case,'nearest',device),'linear':restore_variant(c_checkpoint,case,'linear',device)}
    if a_checkpoint:items={'case_only':restore_variant(a_checkpoint,case,'case_only',device),**items}
    raw_observed=[]
    for raw_margin in (margin_mm,robust_margin_mm):
        rm,rz,ru,ro,rf=items['linear'];rl,rh,rb=bounds(ro,raw_margin)
        for idx,pos in zip(ro['observed'],ro['s'].cpu().numpy()):
            rq=plane_grid(float(pos),rl,rh,rb,resolution)
            lc=query_variant(rm,rq,ro,ru,'linear');ln=query_variant(rm,rq,ro,ru,'nearest')
            raw_observed.append(dict(index=idx,margin_mm=raw_margin,max_abs_logit_diff=float(abs(lc-ln).max()),
                classification_flip_count=int(np.sum((lc>=0)!=(ln>=0)))))
    items={method:promote(item) for method,item in items.items()}
    obs=items['linear'][3];full=items['linear'][4]
    if any(x[3]['observed']!=obs['observed'] for x in items.values()):raise ValueError('ABC observation mismatch')
    if any(not torch.equal(x[3]['s'].cpu(),obs['s'].cpu()) for x in items.values()):raise ValueError('ABC coordinate mismatch')
    before={method:{k:v.detach().cpu().clone() for k,v in x[0].state_dict().items()} for method,x in items.items()}
    if not torch.equal(items['nearest'][1],items['linear'][1]) or not torch.equal(items['nearest'][2],items['linear'][2]):raise AssertionError('BC code/latents differ')
    references=json.loads(Path(reference_metrics).read_text()) if reference_metrics else None
    audit=json.loads(Path(reference_audit).read_text()) if reference_audit else None
    checks=dict(bc_observed=[],c_native_regression=[],checkpoint_read_only=False,same_query_grid=True,threshold_logit=0,
        original_query_used_for_C=True,original_geometry_and_marching_cubes=True,
        raw_float32_bc_observed=raw_observed,evaluation_precision='shared float64 field arithmetic after identical native float32 xyz rounding; fixed native G outputs promoted',
        precision_reason='oblique axis + float32 alpha coefficient rounding otherwise exceeds 1e-5 BC tolerance without classification flips')
    mainimages={};summaries={};csvrows=[];meshes={}
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    for margin in [margin_mm,robust_margin_mm]:
        lower,upper,basis=bounds(obs,margin);rows={m:[] for m in items};images={m:[] for m in items};truths=[]
        heights=[(float(pos),i) for i,pos in enumerate(full['s'].numpy()) if lower[2]-1e-6<=pos<=upper[2]+1e-6]
        heights += [(float((left+right)/2),None) for left,right in zip(obs['s'][:-1].cpu().numpy(),obs['s'][1:].cpu().numpy())]
        qhash=hashlib.sha256()
        for pos,index in heights:
            q=plane_grid(pos,lower,upper,basis,resolution);qhash.update(q.tobytes());predictions={}
            region=contour_labels(q,full['boundary'][index].numpy(),full['boundary_endo'][index].numpy(),basis)[0] if index is not None else None
            for method,(model,code,u,o,_) in items.items():
                logits=query_variant(model,q,o,u,method);predictions[method]=logits
                row,gt=contour_metrics(logits,pos,index,full,obs,lower,upper,basis,resolution,region)
                rows[method].append(row);images[method].append(logits.reshape(resolution,resolution))
                flat=dict(variant=NAMES[method],margin_mm=margin,**{k:v for k,v in row.items() if not isinstance(v,dict)})
                for boundary in ('epi','endo'):
                    for measure in ('mean_signed_distance_pred_to_gt_mm','mean_absolute_distance_pred_to_gt_mm'):
                        flat[boundary+'_'+measure]=row['boundary_error'][boundary][measure] if row['boundary_error'] else None
                csvrows.append(flat)
            truths.append(gt)
            if index in obs['observed']:
                difference=float(abs(predictions['nearest']-predictions['linear']).max());same=bool(np.array_equal(predictions['nearest']>=0,predictions['linear']>=0))
                checks['bc_observed'].append(dict(index=index,margin_mm=margin,max_abs_logit_diff=difference,same_classification=same))
                if difference>1e-5 or not same:raise AssertionError(f'BC observed predictions changed at {index}: {difference}')
                for k in ('contour_proxy_dice','curves','closed_curves','open_curves','positive_components','positive_touches_border'):
                    if rows['nearest'][-1][k]!=rows['linear'][-1][k]:raise AssertionError('BC observed topology differs')
            if index is not None and margin==margin_mm and references:
                ref=next(r for r in references['planes'] if r['cached_plane_index']==index)
                current=rows['linear'][-1];diff=abs(current['contour_proxy_dice']-ref['contour_proxy_dice'])
                if diff>1e-4:raise AssertionError(f'C regression mismatch at plane {index}: {diff}')
                for k in ('curves','closed_curves','open_curves','positive_components','positive_touches_border'):
                    if current[k]!=ref[k]:raise AssertionError(f'C topology regression mismatch: {index} {k}')
                if audit:
                    prior=next(r for r in audit['planes'] if r.get('index',r.get('cached_plane_index'))==index)
                    for key in ('predicted_area_mm2','gt_area_mm2','predicted_to_gt_area_ratio'):
                        if key in prior and abs(current[key]-prior[key])>1e-5:raise AssertionError(f'C area regression mismatch {index} {key}')
                    if current['single_connected_ring']!=prior.get('single_connected_ring',True):raise AssertionError('C ring regression mismatch')
                checks['c_native_regression'].append(dict(index=index,dice_absolute_difference=diff,topology_equal=True))
        label='main' if margin==margin_mm else 'margin20mm'
        summaries[label]={}
        for method in items:
            target=out/NAMES[method] if label=='main' else out/'robust_margin20mm'/NAMES[method];target.mkdir(parents=True,exist_ok=True)
            report=dict(model_variant=method,case_id=obs['case_id'],phase=obs['phase'],observed_indices=obs['observed'],
                threshold_logit=0,resolution=resolution,margin_mm=margin,query_grid_sha256=qhash.hexdigest(),planes=rows[method],
                checkpoint_sha256=digest(a_checkpoint) if method=='case_only' else sourcehash,
                evaluation_precision=checks['evaluation_precision'],
                target='filled Epi minus filled Endo contour proxy, NOT MRI raster',no_extrapolation=True)
            summary={}
            for kind,name in [('observed','observed'),('strict_heldout','heldout')]:
                selected=[r for r in rows[method] if r['kind']==kind]
                summary[name+'_mean_dice']=float(np.mean([r['contour_proxy_dice'] for r in selected]))
                summary[name+'_ring_coverage']=float(np.mean([r['single_connected_ring'] for r in selected]))
                for boundary in ('epi','endo'):
                    values=[r['boundary_error'][boundary]['mean_signed_distance_pred_to_gt_mm'] for r in selected]
                    summary[name+'_'+boundary+'_offset_mm']=float(np.mean(values)) if all(v is not None for v in values) else None
            summary['heldout_minus_observed']=summary['heldout_mean_dice']-summary['observed_mean_dice'];report['summary']=summary
            (target/'metrics.json').write_text(json.dumps(report,indent=2));summaries[label][NAMES[method]]=summary
        if label=='main':
            mainimages=images
            fig,axes=plt.subplots(len(heights),len(items),figsize=(4*len(items),3*len(heights)),squeeze=False)
            for col,method in enumerate(items):
                for row,(pos,index) in enumerate(heights):
                    ax=axes[row,col];im=images[method][row];ax.imshow((im>=0).T,origin='lower',cmap='Blues',vmin=0,vmax=1)
                    if im.min()<0<im.max():ax.contour(im.T,[0],colors=['cyan'],linewidths=.7)
                    if index is not None:
                        for poly,color in [(full['boundary'][index].numpy(),'orange'),(full['boundary_endo'][index].numpy(),'red')]:
                            uv=(poly@basis[:,:2]-lower[:2])/(upper[:2]-lower[:2])*(resolution-1);uv=np.vstack([uv,uv[0]])
                            ax.plot(uv[:,0],uv[:,1],color=color,linewidth=.8)
                    ax.set_xlim(0,resolution-1);ax.set_ylim(0,resolution-1);ax.set_aspect('equal')
                    ax.set_title(f'{NAMES[method]} | {index if index is not None else "gap"} {pos*obs["scale_mm"]:.1f} mm')
            fig.tight_layout();fig.savefig(out/'plane_triptych.png',dpi=120);plt.close(fig)
            jumps=discontinuity(items,obs,lower,upper,basis,resolution);(out/'nearest_discontinuity.json').write_text(json.dumps(jumps,indent=2))
            if with_mesh:
                n=mesh_resolution;grid=[np.linspace(lower[i],upper[i],n) for i in range(3)];xx,yy=np.meshgrid(grid[0],grid[1],indexing='ij')
                qmeshhash=hashlib.sha256()
                volumes={m:np.empty((n,n,n),np.float32) for m in items}
                for i,pos in enumerate(grid[2]):
                    q=np.stack([xx.ravel(),yy.ravel(),np.full(xx.size,pos)],-1)@basis.T;qmeshhash.update(q.tobytes())
                    for method,(model,code,u,o,_) in items.items():volumes[method][:,:,i]=query_variant(model,q,o,u,method).reshape(n,n)
                for method,volume in volumes.items():
                    report,arrays=mesh_report(volume,lower,upper,basis,obs,out/NAMES[method]/'myocardium.obj');meshes[method]=arrays
                    if method=='linear' and reference_mesh and n==128 and margin==5.:
                        prior=json.loads(Path(reference_mesh).read_text())
                        for k in ('vertices','faces','boundary_edges','nonmanifold_edges','vertices_on_lateral_boundary'):
                            if report[k]!=prior[k]:raise AssertionError(f'C native mesh regression mismatch {k}: {report[k]} != {prior[k]}')
                        checks['c_native_mesh_regression']=True
                checks['mesh_query_grid_sha256']=qmeshhash.hexdigest()
                fig=plt.figure(figsize=(5*len(items),5));from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                worldcorners=np.array([[x,y,z] for x in (lower[0],upper[0]) for y in (lower[1],upper[1]) for z in (lower[2],upper[2])])@basis.T*obs['scale_mm']+obs['center_mm']
                lo=worldcorners.min(0);hi=worldcorners.max(0)
                for col,method in enumerate(items):
                    ax=fig.add_subplot(1,len(items),col+1,projection='3d');arrays=meshes[method]
                    if arrays:
                        v,f=arrays;faces=f[np.linspace(0,len(f)-1,min(len(f),16000),dtype=int)]
                        ax.add_collection3d(Poly3DCollection(v[faces],facecolor='#65a9cd',edgecolor='none',alpha=.9))
                    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(hi-lo);ax.view_init(18,35);ax.set_title(NAMES[method])
                fig.tight_layout();fig.savefig(out/'mesh_triptych.png',dpi=150);plt.close(fig)
    for method,x in items.items():
        if not all(torch.equal(v.detach().cpu(),before[method][k]) for k,v in x[0].state_dict().items()):raise AssertionError('Inference mutated model')
    if digest(c_checkpoint)!=sourcehash:raise AssertionError('C checkpoint changed')
    checks['checkpoint_read_only']=True
    fields=sorted(set().union(*(r.keys() for r in csvrows)))
    with (out/'metrics_all_planes.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(csvrows)
    (out/'comparison_summary.json').write_text(json.dumps(summaries,indent=2));(out/'validation_checks.json').write_text(json.dumps(checks,indent=2))
    return summaries,checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--a-checkpoint');p.add_argument('--c-checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--resolution',type=int,default=160);p.add_argument('--mesh-resolution',type=int,default=128)
    p.add_argument('--margin-mm',type=float,default=5.);p.add_argument('--robust-margin-mm',type=float,default=20.)
    p.add_argument('--reference-metrics');p.add_argument('--reference-mesh');p.add_argument('--reference-audit');p.add_argument('--no-mesh',action='store_true')
    a=p.parse_args();summary,checks=evaluate(a.case,a.c_checkpoint,a.output,a.a_checkpoint,a.device,a.resolution,a.mesh_resolution,a.margin_mm,
        a.robust_margin_mm,a.reference_metrics,a.reference_mesh,a.reference_audit,not a.no_mesh);print(json.dumps(dict(summary=summary,validation=checks),indent=2))

if __name__=='__main__':main()
