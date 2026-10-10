"""Fixed split evaluation and honest mesh/coverage reporting for a single epi field."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.data import load_case
from core.model import PlaneField,leave_one_out_latent,interpolate_latent
from core.losses import coordinate_gradient
from core.evaluation import case_geometry,aggregate_geometry,slice_geometry
from extract_mesh import extract


def diagnostics(model,c,out,foreign=None):
    with torch.no_grad():
        z=model.encoder(c['q'],c['sdf2d'],c['phase']);zl=leave_one_out_latent(c['s'],z)
        prob=torch.softmax(model.head(z)/0.1,-1)
    probes=0.5*(c['boundary'][:-1,::8]+c['boundary'][1:,::8])
    q=probes.reshape(-1,3).detach().requires_grad_(True)
    f=model.field(q,c['s'],z,c['axis'],c['phase'])
    norm=coordinate_gradient(f,q,False).norm(dim=-1).detach().cpu().numpy()
    sign_groups=[]
    for i in range(len(z)):
        with torch.no_grad():
            pred=model.decoder(c['q'][i],z[i:i+1].expand(c['q'].shape[1],-1),c['phase'])
        d=c['sdf2d'][i];mask=d.abs()>0.002
        groups=[(pred[mask&(d<0)]<0).float(),(pred[mask&(d>0)]>0).float()]
        sign_groups.append(float(torch.stack([x.mean() for x in groups if len(x)]).mean()))
    chosen=sorted(set([0,len(z)//2,len(z)-1]))
    fig,axes=plt.subplots(2,len(chosen),figsize=(4*len(chosen),8),squeeze=False)
    from core.mesh import stack_basis
    basis=stack_basis(c['axis'].cpu().numpy())
    for j,i in enumerate(chosen):
        for row,latent in enumerate([z[i:i+1],zl[i-1:i] if 0<i<len(z)-1 else z[i:i+1]]):
            _,field,lower,step=slice_geometry(model,c,latent,i,1.0)
            ax=axes[row,j]
            extent=[lower[0]*100,(lower[0]+step*(field.shape[0]-1))*100,lower[1]*100,(lower[1]+step*(field.shape[1]-1))*100]
            ax.imshow(field.T,origin='lower',extent=extent,cmap='coolwarm',vmin=-.03,vmax=.03)
            if field.min()<0<field.max():
                ax.contour(np.linspace(extent[0],extent[1],field.shape[0]),np.linspace(extent[2],extent[3],field.shape[1]),field.T,levels=[0],colors=['lime'])
            b=c['boundary'][i].cpu().numpy()@basis[:,:2]*100;b=np.r_[b,b[:1]]
            ax.plot(b[:,0],b[:,1],'k--',linewidth=1)
            ax.set_title(f'{"direct" if row==0 else "LOO" if 0<i<len(z)-1 else "endpoint (direct)"}, plane {i}')
            ax.set_aspect('equal')
    fig.suptitle(f'{c["case_id"]} {"ED" if c["phase"]==0 else "ES"}: black observation, green zero contour')
    fig.tight_layout();fig.savefig(out/'contours.png',dpi=150);plt.close(fig)
    result=dict(sign_balanced_accuracy=float(np.mean(sign_groups)),gap_gradient_norm_quantiles=np.quantile(norm,[0,.05,.5,.95,1]).tolist(),
                latent_std=float(z.std(0,unbiased=False).mean()),student_entropy=float(-(prob*prob.clamp_min(1e-9).log()).sum(-1).mean()))
    if foreign is not None:
        with torch.no_grad():
            other=model.encoder(foreign['q'],foreign['sdf2d'],foreign['phase'])
            fraction=(c['s']-c['s'][0])/(c['s'][-1]-c['s'][0])
            positions=foreign['s'][0]+fraction*(foreign['s'][-1]-foreign['s'][0])
            queries=positions[:,None]*foreign['axis']
            swapped=interpolate_latent(queries,foreign['s'],other,foreign['axis'])
            probes=c['q'][:,::16].reshape(-1,3)
            original=model.field(probes,c['s'],z,c['axis'],c['phase'])
            changed=model.field(probes,c['s'],swapped,c['axis'],c['phase'])
            result['swapped_case_id']=foreign['case_id']
            result['condition_swap_field_change_mm']=float((original-changed).abs().mean()*100)
            result['condition_latent_change']=float((z-swapped).square().mean().sqrt())
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['train','val','test'],default='test');p.add_argument('--device',default='cpu')
    p.add_argument('--resolution',type=int,default=160);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--meshes',action='store_true');p.add_argument('--spacing-mm',type=float,default=1.0)
    args=p.parse_args();torch.set_num_threads(1)
    saved=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    model=PlaneField(**saved['config']['model']).to(args.device).eval();model.load_state_dict(saved['student'])
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    paths=sorted((Path(args.data)/args.split).glob('*.npz'))
    if args.limit:paths=paths[:args.limit]
    all_rows, failures, csv_rows=[],[],[]
    references={}
    for path in paths:
        c=load_case(path,args.device);name=path.stem;out=root/name;out.mkdir(exist_ok=True)
        geometry=case_geometry(model,c,args.spacing_mm)
        diag=diagnostics(model,c,out,references.get(c['phase']))
        references.setdefault(c['phase'],c)
        # Slab descriptors only: no unknown end completion or cavity/EF claim.
        sections=geometry['direct'];ds=np.diff([r['s_mm'] for r in sections])
        observed=np.asarray([r['observed_section_area_mm2'] for r in sections])
        diag['observed_slab_volume_descriptor_ml']=float(np.sum((observed[:-1]+observed[1:])/2*ds)/1000)
        if all(r['predicted_section_area_mm2'] is not None for r in sections):
            predicted=np.asarray([r['predicted_section_area_mm2'] for r in sections])
            diag['predicted_slab_volume_descriptor_ml']=float(np.sum((predicted[:-1]+predicted[1:])/2*ds)/1000)
        else:
            diag['predicted_slab_volume_descriptor_ml']=None
        report=dict(case_id=c['case_id'],phase=c['phase'],split=args.split,geometry=geometry,diagnostics=diag)
        if args.meshes:
            try:report['mesh']=extract(model,c,out/'epi.obj',args.resolution)
            except ValueError as e:
                report['mesh_failure']=str(e);failures.append(dict(case=name,error=str(e)))
        (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        all_rows.append(geometry)
        csv_rows.append(dict(case_id=c['case_id'],phase=c['phase'],**aggregate_geometry([geometry]),**{k:diag.get(k) for k in ['sign_balanced_accuracy','latent_std','student_entropy','swapped_case_id','condition_swap_field_change_mm','condition_latent_change','observed_slab_volume_descriptor_ml','predicted_slab_volume_descriptor_ml']}))
        print(json.dumps(dict(case=name,**csv_rows[-1])),flush=True)
    if not all_rows:raise ValueError('Empty evaluation split')
    with (root/'phase_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=csv_rows[0].keys());writer.writeheader();writer.writerows(csv_rows)
    summary=dict(split=args.split,patient_count=len(set(r['case_id'] for r in csv_rows)),phase_file_count=len(paths),
                 evaluation_query_spacing_mm=args.spacing_mm,mesh_resolution=args.resolution if args.meshes else None,
                 checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_epoch=saved['epoch'],metrics=aggregate_geometry(all_rows),mesh_failures=failures,
                 metric_contract='LV epi filled envelope, SAX-only; same 0.5mm raster Dice and symmetric point-sampled MCD/HD95 as v11_5static. Direct and real layer LOO separated. Phase then patient equal weight.',
                 distances_are_proxy=False,full_3d_gt_metrics_available=False,cardiac_function_reported=False,artificial_caps=False)
    (root/'numeric_eval_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    # Marker means all phases assessed; failures remain explicit in the summary.
    (root/'EVALUATION_COMPLETE.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
