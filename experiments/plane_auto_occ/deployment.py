"""Cold-start two-step smoke and independent single ED fit; evaluate at the end."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch


def verify_sources(exp):
    root=Path(__file__).resolve().parents[2]
    manifest=json.loads((exp/'source_lock.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Locked source changed: {name}')
    cache=exp/'cache/train/002_ED.npz'
    if hashlib.sha256(cache.read_bytes()).hexdigest()!=manifest['cache_sha256']:
        raise ValueError('Locked cache changed')
    return cache,root,manifest


def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',required=True)
    p.add_argument('--device',default='cuda'); a=p.parse_args()
    exp=Path(a.experiment).resolve(); case,root,lock=verify_sources(exp)
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    def run(module,*args):
        subprocess.run([sys.executable,'-B','-m','experiments.plane_auto_occ.'+module,*map(str,args)],cwd=root,check=True)
    try:
        run('tests')
        smoke=exp/'outputs/cuda_smoke'
        run('train','--case',case,'--observed','0,3,6','--output',smoke,'--device',a.device,
            '--steps',2,'--milestones','0,2','--diagnostics-resolution',64)
        ck=torch.load(smoke/'last.pt',map_location='cpu',weights_only=False)
        assert ck['step']==2 and ck['records'][0]['observed']==[0,3,6] and ck['mode']=='joint'
        assert all(torch.isfinite(v).all() for v in ck['model'].values()) and torch.isfinite(ck['codes']).all()
        (exp/'gates/GPU_SMOKE_COMPLETE.json').write_text(json.dumps(dict(device=a.device,steps=2,
            cuda_name=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
            torch_version=torch.__version__,source_commit=lock['commit']),indent=2))
        fit=exp/'outputs/fit'
        # Fresh process and seed reset: no smoke weights, optimizer, or RNG reused.
        run('train','--case',case,'--observed','0,3,6','--output',fit,'--device',a.device,
            '--steps',2000,'--milestones','0,50,100,200,500,1000,2000')
        ck=torch.load(fit/'last.pt',map_location='cpu',weights_only=False)
        assert ck['step']==2000 and ck['records'][0]['observed']==[0,3,6]
        (exp/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(steps=2000,checkpoint=str(fit/'last.pt'),
            observed=[0,3,6],strict_heldout=[1,2,4,5],source_commit=lock['commit']),indent=2))
        # Formal heldout geometry evaluation starts only after training completes.
        evaluation=exp/'outputs/final_eval'
        run('evaluate','--checkpoint',fit/'last.pt','--case',case,'--output',evaluation,'--device',a.device)
        metrics=json.loads((evaluation/'metrics.json').read_text())
        assert [r['cached_plane_index'] for r in metrics['planes'] if r['kind']=='strict_heldout']==[1,2,4,5]
        margin=5
        if any(r.get('ground_truth_touches_border',False) for r in metrics['planes']):
            margin=20; evaluation=exp/'outputs/final_eval_margin20mm'
            run('evaluate','--checkpoint',fit/'last.pt','--case',case,'--output',evaluation,
                '--device',a.device,'--margin-mm',margin)
            metrics=json.loads((evaluation/'metrics.json').read_text())
        mesh=exp/'outputs/myocardium.obj'
        result=subprocess.run([sys.executable,'-B','-m','experiments.plane_auto_occ.extract_mesh',
            '--checkpoint',str(fit/'last.pt'),'--case',str(case),'--output',str(mesh),
            '--device',a.device,'--resolution','128','--margin-mm',str(margin)],cwd=root,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        (exp/'logs/mesh_stdout.log').write_text(result.stdout)
        (exp/'logs/mesh_stderr.log').write_text(result.stderr)
        if result.returncode:
            if 'No finite logit zero crossing; cannot extract mesh' not in result.stderr:
                raise RuntimeError(result.stderr)
            mesh.with_suffix('.json').write_text(json.dumps(dict(status='no_zero_crossing'),indent=2))
        complete=dict(steps=2000,evaluation=str(evaluation),mesh_extracted=mesh.exists(),
            heldout_ground_truth_touches_border=any(r.get('ground_truth_touches_border',False) for r in metrics['planes']),
            source_commit=lock['commit'],scope='single-case execution complete; inspect geometry and latent sensitivity')
        (exp/'RUN_COMPLETE.json').write_text(json.dumps(complete,indent=2)); print(json.dumps(complete),flush=True)
    except Exception as exc:
        (exp/'PIPELINE_FAILED.json').write_text(json.dumps(dict(type=type(exc).__name__,error=str(exc)),indent=2))
        raise


if __name__=='__main__': main()
