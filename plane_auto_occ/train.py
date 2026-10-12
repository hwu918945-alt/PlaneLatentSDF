"""Joint fitting or frozen-shared-decoder new-case latent fitting."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .model import AutoOccupancy
from .data import load_case, records
from .objective import objective


def parser():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--data-root'); g.add_argument('--case'); g.add_argument('--manifest')
    p.add_argument('--observed', help='zero-based cached plane indices, e.g. 0,3,6')
    p.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--output',required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int); p.add_argument('--pretrained',help='freeze shared model; optimize NEW codes only')
    p.add_argument('--milestones', help='Single-case read-only observed-plane diagnostics, e.g. 0,50,100,200,500,1000,2000')
    p.add_argument('--diagnostics-resolution', type=int, default=96)
    p.add_argument('--area-pool',help='Observed-only area-uniform V2 pool')
    return p


def main():
    a=parser().parse_args(); cfg=json.loads(Path(a.config).read_text())
    if a.steps is not None: cfg['steps']=a.steps
    if cfg.get('pipeline_backend')=='paired_roi_exact_v1':
        raise ValueError('Use plane_auto_occ.paired_suite: legacy train --area-pool has no ROI-source contract')
    if cfg.get('sampling_version')=='observed_area_uniform_v2' and not a.area_pool:
        raise ValueError('V2 requires --area-pool; legacy near/far sampling cannot be used')
    milestones=set(int(x) for x in a.milestones.split(',')) if a.milestones else set()
    if milestones and (not a.case or min(milestones)<0 or max(milestones)>cfg['steps']):
        raise ValueError('Milestones require --case and steps within the training budget')
    if cfg['steps'] < 1 or cfg['points'] < 4 or cfg['points'] % 4: raise ValueError('Invalid steps/points')
    if min(cfg['threads'],cfg['log_every'],cfg['save_every'],cfg['clip_grad'],cfg['lr_shared'],cfg['lr_latent']) <= 0:
        raise ValueError('Threads, intervals, clip norm and learning rates must be positive')
    if cfg['seam_eps_mm'] <= 0 or min(cfg['direct_weight'],cfg['loo_weight'],cfg['latent_weight'],cfg['seam_weight']) < 0:
        raise ValueError('Invalid loss config')
    torch.set_num_threads(cfg['threads']); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Output must be new/empty; no implicit resume')
    out.mkdir(parents=True,exist_ok=True)
    observed=[int(x) for x in a.observed.split(',')] if a.observed else None
    if a.manifest and observed is not None: raise ValueError('Put observed indices in manifest')
    rows=records(a.data_root,a.case,a.manifest,observed)
    # Keep large point pools on CPU; transfer only current case.
    cases=[load_case(p,observed=ids) for p,ids in rows]
    keys=[(c['case_id'],c['phase']) for c in cases]
    if len(set(keys)) != len(keys): raise ValueError('Duplicate case/phase files')
    if cfg['loo_weight'] > 0 and any(len(c['s']) < 3 for c in cases):
        raise ValueError('LOO needs >=3 observed planes; set loo_weight=0 for two-plane inference')
    model=AutoOccupancy(cfg['latent_dim'],cfg['width']).to(a.device)
    prior=None
    if a.pretrained:
        prior=torch.load(a.pretrained,map_location='cpu',weights_only=False)
        if prior['format'] != 'plane_auto_occ_v1': raise ValueError('Wrong checkpoint format')
        for k in ('latent_dim','width'):
            if cfg[k] != prior['config'][k]: raise ValueError(f'{k} must match pretrained checkpoint')
        model.load_state_dict(prior['model']); model.requires_grad_(False)
        previous={tuple(k) for k in prior['training_keys']}
        if previous.intersection(keys): raise ValueError('New-case inference includes training patient/phase')
        train_patients={k[0] for k in previous}
        if train_patients.intersection(k[0] for k in keys): raise ValueError('Patient leakage across ED/ES')
    codes=torch.nn.Parameter(torch.randn(len(cases),cfg['latent_dim'],device=a.device)*0.01)
    groups=[dict(params=[codes],lr=cfg['lr_latent'])]
    if not a.pretrained: groups.append(dict(params=model.parameters(),lr=cfg['lr_shared']))
    opt=torch.optim.Adam(groups)
    if a.area_pool:
        from .area_pool import AreaSampler,digest,build_pool
        from .chunked import backward_case
        if len(cases)!=1: raise ValueError('This repair run is a single case')
        sampler=AreaSampler(a.area_pool,cfg['seed'])
        if sampler.meta['observed']!=cases[0]['observed'] or sampler.meta['source_sha256']!=digest(cases[0]['path']):
            raise ValueError('Pool observation/source mismatch')
        cases[0]['_area_sampler']=sampler
    manifest=[]
    for c in cases:
        manifest.append(dict(path=c['path'],case_id=c['case_id'],phase=c['phase'],observed=c['observed'],
                             source_sha256=hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()))
    (out/'fit_manifest.json').write_text(json.dumps(dict(records=manifest,mode='latent_only' if a.pretrained else 'joint',
                                      pretrained=a.pretrained,config=cfg),indent=2))
    (out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
    train_keys=prior['training_keys'] if prior else [list(k) for k in keys]
    def save(step):
        payload=dict(format='plane_auto_occ_v1',model=model.state_dict(),codes=codes.detach().cpu(),config=cfg,
                     records=manifest,training_keys=train_keys,mode='latent_only' if a.pretrained else 'joint',step=step)
        if a.area_pool:
            payload['sampling_pool']=dict(path=sampler.path,sha256=digest(sampler.path),generation=sampler.meta['generation'])
            payload['objective_version']=cfg['objective_version']
        tmp=out/'last.pt.tmp'; torch.save(payload,tmp); tmp.replace(out/'last.pt')
    def diagnose(step):
        if step not in milestones: return
        import shutil
        save(step)
        dest=out/f'step_{step:05d}'; dest.mkdir()
        checkpoint=dest/'checkpoint.pt'; shutil.copyfile(out/'last.pt',checkpoint)
        # Separate process: diagnostics cannot advance training RNG or change parameters.
        subprocess.run([sys.executable,'-B','-m','plane_auto_occ.diagnostics',
                        '--checkpoint',str(checkpoint),'--case',a.case,'--output',str(dest/'diagnostics'),
                        '--device',a.device,'--resolution',str(a.diagnostics_resolution)],check=True)
    diagnose(0)
    order=[]
    with (out/'history.jsonl').open('w') as log:
        for step in range(cfg['steps']):
            update_start=time.perf_counter()
            if a.area_pool and step and step%cfg['pool_refresh_updates']==0:
                generation=step//cfg['pool_refresh_updates']
                target=out/'sampling_pools'/f'generation_{generation:03d}.npz'
                build_pool(cases[0]['path'],cases[0]['observed'],target,tuple(cfg['pool_capacities']),cfg['seed'],generation,
                           cfg['pool_margin_mm'],cfg['boundary_band_mm'],cfg['near_mm'])
                sampler=AreaSampler(target,cfg['seed']); cases[0]['_area_sampler']=sampler
            if not order: order=torch.randperm(len(cases)).tolist()
            idx=order.pop()
            active={'axis','s','boundary','boundary_endo'} if a.area_pool else set(cases[idx])
            c={k:(v.to(a.device) if k in active and torch.is_tensor(v) else v) for k,v in cases[idx].items()}
            opt.zero_grad(set_to_none=True)
            if a.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
            loss,stats=backward_case(model,codes[idx],c,cfg,step) if a.area_pool else objective(model,codes[idx],c,cfg,step)
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f'Nonfinite loss at step {step}')
            if not a.area_pool: loss.backward()
            params=[codes]+([] if a.pretrained else list(model.parameters()))
            norm=torch.nn.utils.clip_grad_norm_(params,cfg['clip_grad'],error_if_nonfinite=True)
            opt.step()
            if a.device.startswith('cuda'): torch.cuda.synchronize()
            row=dict(step=step+1,case_id=c['case_id'],phase=c['phase'],loss=float(loss.detach()),gradient_norm=float(norm),**stats)
            if a.area_pool:
                row.update(update_seconds=time.perf_counter()-update_start,pool_generation=sampler.meta['generation'],
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0,
                           peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if a.device.startswith('cuda') else 0)
            log.write(json.dumps(row)+'\n'); log.flush()
            if step == 0 or (step+1)%cfg['log_every'] == 0: print(json.dumps(row),flush=True)
            if (step+1)%cfg['save_every'] == 0: save(step+1)
            diagnose(step+1)
    save(cfg['steps'])
    print(f'Saved {out / "last.pt"}; evaluate geometry separately')


if __name__ == '__main__': main()

