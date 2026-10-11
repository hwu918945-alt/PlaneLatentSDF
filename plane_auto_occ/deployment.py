"""Cold-start two-step smoke and independent single ED fit; evaluate at the end."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch


def verify_sources(exp):
    exp=Path(exp).resolve()
    root=exp/'runtime' if (exp/'runtime').is_dir() else Path(__file__).resolve().parents[1]
    manifest=json.loads((exp/'source_lock.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Locked source changed: {name}')
    cache=exp/manifest.get('cache_path','cache/train/002_ED.npz')
    if hashlib.sha256(cache.read_bytes()).hexdigest()!=manifest['cache_sha256']:
        raise ValueError('Locked cache changed')
    if 'pool_sha256' in manifest:
        if hashlib.sha256((exp/'pools/generation_000.npz').read_bytes()).hexdigest()!=manifest['pool_sha256']:
            raise ValueError('Initial area pool changed')
    for prefix in ('config','reference_initial'):
        if prefix+'_sha256' in manifest:
            if hashlib.sha256((exp/manifest[prefix+'_path']).read_bytes()).hexdigest()!=manifest[prefix+'_sha256']:
                raise ValueError(f'Locked {prefix} changed')
    return cache,root,manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',required=True)
    p.add_argument('--device',default='cuda'); a=p.parse_args()
    exp=Path(a.experiment).resolve(); case,root,lock=verify_sources(exp)
    config=exp/lock['config_path'] if 'config_path' in lock else Path(__file__).with_name('config.json')
    cfg=json.loads(config.read_text()); steps=int(cfg['steps'])
    observed=lock.get('observed',[0,3,6]); observed_arg=','.join(map(str,observed))
    from .data import load_case
    nplanes=load_case(case,observed=observed)['total_planes']
    heldout=sorted(set(range(nplanes))-set(observed))
    milestones=sorted({v for v in [0,50,100,200,500,1000,2000,steps] if v<=steps})
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    def run(module,*args):
        subprocess.run([sys.executable,'-B','-m','plane_auto_occ.'+module,*map(str,args)],cwd=root,check=True)
    try:
        run('tests')
        run('repair_tests')
        pool=exp/'pools/generation_000.npz'
        smoke=exp/'outputs/cuda_smoke'
        run('train','--case',case,'--observed',observed_arg,'--config',config,'--area-pool',pool,'--output',smoke,'--device',a.device,
            '--steps',2,'--milestones','0,2','--diagnostics-resolution',64)
        ck=torch.load(smoke/'last.pt',map_location='cpu',weights_only=False)
        assert ck['step']==2 and ck['records'][0]['observed']==observed and ck['mode']=='joint'
        assert all(torch.isfinite(v).all() for v in ck['model'].values()) and torch.isfinite(ck['codes']).all()
        baseline=exp/lock['reference_initial_path'] if 'reference_initial_path' in lock else None
        if baseline is not None:
            initial=torch.load(smoke/'step_00000/checkpoint.pt',map_location='cpu',weights_only=False)
            reference=torch.load(baseline,map_location='cpu',weights_only=False)
            assert all(torch.equal(initial['model'][k],v) for k,v in reference['model'].items())
            assert torch.equal(initial['codes'],reference['codes'])
        (exp/'gates/GPU_SMOKE_COMPLETE.json').write_text(json.dumps(dict(device=a.device,steps=2,
            cuda_name=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
            torch_version=torch.__version__,source_commit=lock['commit']),indent=2))
        fit=exp/'outputs/fit'
        # Fresh process and seed reset: no smoke weights, optimizer, or RNG reused.
        run('train','--case',case,'--observed',observed_arg,'--config',config,'--area-pool',pool,'--output',fit,'--device',a.device,
            '--steps',steps,'--milestones',','.join(map(str,milestones)))
        ck=torch.load(fit/'last.pt',map_location='cpu',weights_only=False)
        assert ck['step']==steps and ck['records'][0]['observed']==observed
        (exp/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(steps=steps,checkpoint=str(fit/'last.pt'),
            observed=observed,strict_heldout=heldout,source_commit=lock['commit']),indent=2))
        # Formal heldout geometry evaluation starts only after training completes.
        evaluation=exp/'outputs/final_eval'
        run('evaluate','--checkpoint',fit/'last.pt','--case',case,'--output',evaluation,'--device',a.device)
        metrics=json.loads((evaluation/'metrics.json').read_text())
        assert [r['cached_plane_index'] for r in metrics['planes'] if r['kind']=='strict_heldout']==heldout
        # The repair protocol always adds a separate 20-mm robustness audit.
        run('evaluate','--checkpoint',fit/'last.pt','--case',case,'--output',exp/'outputs/final_eval_margin20mm',
            '--device',a.device,'--margin-mm',20)
        margin=5
        mesh=exp/'outputs/myocardium.obj'
        result=subprocess.run([sys.executable,'-B','-m','plane_auto_occ.extract_mesh',
            '--checkpoint',str(fit/'last.pt'),'--case',str(case),'--output',str(mesh),
            '--device',a.device,'--resolution','128','--margin-mm',str(margin)],cwd=root,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        (exp/'logs/mesh_stdout.log').write_text(result.stdout)
        (exp/'logs/mesh_stderr.log').write_text(result.stderr)
        if result.returncode:
            if 'No finite logit zero crossing; cannot extract mesh' not in result.stderr:
                raise RuntimeError(result.stderr)
            mesh.with_suffix('.json').write_text(json.dumps(dict(status='no_zero_crossing'),indent=2))
        complete=dict(steps=steps,evaluation=str(evaluation),mesh_extracted=mesh.exists(),
            heldout_ground_truth_touches_border=any(r.get('ground_truth_touches_border',False) for r in metrics['planes']),
            source_commit=lock['commit'],scope='single-case execution complete; inspect geometry and latent sensitivity')
        import numpy as np
        from .inference import restore,bounds,plane_grid,query
        from .geometry import contour_labels
        from .diagnostics import topology
        model,code,u,obs,full=restore(fit/'last.pt',case,a.device)
        lower,upper,basis=bounds(obs,5);rows=[]
        for i in range(len(full['s'])):
            pos=float(full['s'][i]); q=plane_grid(pos,lower,upper,basis,160)
            logits=query(model,q,obs,u); pred=logits>=0
            region,_,_=contour_labels(q,full['boundary'][i].numpy(),full['boundary_endo'][i].numpy(),basis)
            gt=region==0
            rows.append(dict(index=i,observed=i in obs['observed'],dice=float((2*(pred & gt).sum()+1e-6)/(pred.sum()+gt.sum()+1e-6)),
                myocardium_recall=float(pred[region==0].mean()),cavity_false_positive_rate=float(pred[region==1].mean()),
                background_false_positive_rate=float(pred[region==2].mean()),**topology(logits.reshape(160,160))))
        complete['posthoc_observed_acceptance']=all(r['dice']>=.90 and r['single_connected_ring'] for r in rows if r['observed'])
        # Report actual mesh open-edge locations without padding or repair.
        if mesh.exists():
            vertices=[];faces=[]
            for line in mesh.read_text().splitlines():
                if line.startswith('v '):vertices.append([float(x) for x in line.split()[1:]])
                elif line.startswith('f '):faces.append([int(x)-1 for x in line.split()[1:]])
            v=np.array(vertices);f=np.array(faces)
            e=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
            e,n=np.unique(e,axis=0,return_counts=True);edge=e[n==1]
            stack=(v-obs['center_mm'])/obs['scale_mm'] @ basis;ends=stack[edge]
            axial=np.any(np.all(np.abs(ends[:,:,2,None]-np.array([lower[2],upper[2]]))<1e-6,axis=1),axis=1)
            lateral=np.any(np.all((np.abs(ends[:,:,:2]-lower[:2])<1e-6)|(np.abs(ends[:,:,:2]-upper[:2])<1e-6),axis=1),axis=1)
            complete['mesh_boundary_edges']=dict(axial=int(axial.sum()),lateral=int(lateral.sum()),other=int((~(axial|lateral)).sum()))
        (exp/'outputs/posthoc_audit.json').write_text(json.dumps(dict(planes=rows,execution=complete),indent=2))
        (exp/'RUN_COMPLETE.json').write_text(json.dumps(complete,indent=2)); print(json.dumps(complete),flush=True)
    except Exception as exc:
        (exp/'PIPELINE_FAILED.json').write_text(json.dumps(dict(type=type(exc).__name__,error=str(exc)),indent=2))
        raise


if __name__=='__main__': main()
