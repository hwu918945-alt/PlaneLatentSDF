"""Case-wise shared training with independent Adam latent state and exact resume.

The unit of a step remains one complete case/phase, with every observed plane,
fixed class budgets, direct/LOO mixture, one joint clip and one Adam update.
"""
import hashlib
import json
import os
import random
import signal
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from .area_pool import AreaSampler,digest
from .shell_pool import build_pool as shell_pool
from .chunked import backward_case
from .data import load_case
from .model import AutoOccupancy


class VisitSampler(AreaSampler):
    """Random access by case visit: stable permutations survive LRU and resume."""
    def set_visit(self,visit,points,seed):
        self.visit=int(visit); self.seed=int(seed); self.points=int(points)

    def draw(self,i,count):
        if count!=self.points: raise ValueError('Visit sampler point budget changed')
        budgets=[count//2,count//4,count//4]
        for r,n in enumerate(budgets):
            cap=self.capacities[r]
            if cap%n or n>cap: raise ValueError('Invalid pool capacity')
            draws=cap//n; cycle=self.visit//draws; offset=(self.visit%draws)*n; key=(i,r)
            if getattr(self,'cycle_keys',{}).get(key)!=cycle:
                rng=np.random.default_rng(np.random.SeedSequence([self.seed,99173,self.meta['generation'],i,r,cycle]))
                self.order[key]=rng.permutation(cap)
                if not hasattr(self,'cycle_keys'): self.cycle_keys={}
                self.cycle_keys[key]=cycle
            self.offset[key]=offset; self.cycles[key]=cycle
        return super().draw(i,count)


class PoolStore:
    def __init__(self,root,cfg,max_loaded=2):
        self.root=Path(root); self.cfg=cfg; self.cache=OrderedDict(); self.max_loaded=max_loaded

    def get(self,c,visits):
        cfg=self.cfg; generation=visits//cfg['pool_refresh_updates']
        key=hashlib.sha256(f"{c['case_id']}:{c['phase']}".encode()).hexdigest()[:16]
        path=self.root/key/f'generation_{generation:04d}.npz'
        cachekey=str(path)
        if cachekey in self.cache:
            sampler=self.cache.pop(cachekey); self.cache[cachekey]=sampler
        else:
            if not path.exists():
                if c.get('schema')=='plane_auto_occ_solid_v1':
                    from .solid import build_pool
                else: build_pool=shell_pool
                case_seed=(cfg['seed']+int(key[:8],16))%(2**32)
                build_pool(c['path'],c['observed'],path,tuple(cfg['pool_capacities']),case_seed,generation,
                    cfg['pool_margin_mm'],cfg['boundary_band_mm'],cfg['near_mm'])
                # Repeated population visits should not pay ZIP decompression on every update.
                # Keep exactly the same arrays and hash, stored uncompressed locally.
                with np.load(path,allow_pickle=False) as z: arrays={k:np.array(z[k]) for k in z.files}
                tmp=path.with_suffix('.raw.npz'); np.savez(tmp,**arrays); tmp.replace(path)
            sampler=VisitSampler(path,cfg['seed'])
            if sampler.meta['source_sha256']!=digest(c['path']) or sampler.meta['observed']!=c['observed']:
                raise ValueError('Pool source/observation mismatch')
            self.cache[cachekey]=sampler
            while len(self.cache)>self.max_loaded: self.cache.popitem(last=False)
            # Checkpoints store visit counters, so prior pools are deterministically
            # reconstructible; keep their small audit JSON, only current NPZ on disk.
            for old in path.parent.glob('generation_*.npz'):
                if old!=path:
                    self.cache.pop(str(old),None); old.unlink()
        sampler.set_visit(visits%cfg['pool_refresh_updates'],cfg['points'],cfg['seed'])
        return sampler


def independent_codes(count,latent_dim,device):
    return torch.nn.ParameterList([torch.nn.Parameter(torch.randn(latent_dim,device=device)*.01) for _ in range(count)])


def atomic_save(payload,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp'); torch.save(payload,temporary); temporary.replace(path)


def fit(paths,observed,cfg,output,device='cpu',steps=None,pretrained=None,resume=False,
        milestones=(),diagnose=None,on_epoch=None,stop_requested=None):
    """Run/resume a split or a single case. Never fit heldout keys jointly."""
    out=Path(output); steps=int(steps if steps is not None else cfg['steps'])
    if steps<1: raise ValueError('Positive training budget required')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()) and not resume: raise FileExistsError(f'Use an empty fit output: {out}')
    torch.set_num_threads(cfg['threads']); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    cases=[load_case(p,observed=ids) for p,ids in zip(paths,observed)]
    if len(cases)!=len(paths): raise ValueError('Observation list mismatch')
    keys=[(c['case_id'],c['phase']) for c in cases]
    if len(set(keys))!=len(keys) or any(len(c['s'])<3 for c in cases): raise ValueError('Duplicate key or fewer than three planes')
    manifest=[dict(path=c['path'],case_id=c['case_id'],phase=c['phase'],observed=c['observed'],source_sha256=digest(c['path'])) for c in cases]
    model=AutoOccupancy(cfg['latent_dim'],cfg['width']).to(device)
    prior=torch.load(pretrained,map_location='cpu',weights_only=False) if pretrained else None
    if prior:
        if prior['format']!='plane_auto_occ_v1': raise ValueError('Checkpoint schema mismatch')
        if any(cfg[k]!=prior['config'][k] for k in ('latent_dim','width')): raise ValueError('Architecture mismatch')
        if {k[0] for k in prior['training_keys']}.intersection(k[0] for k in keys): raise ValueError('Patient leakage')
        model.load_state_dict(prior['model']); model.requires_grad_(False)
    codes=independent_codes(len(cases),cfg['latent_dim'],device)
    groups=[dict(params=codes.parameters(),lr=cfg['lr_latent'])]
    if not prior: groups.append(dict(params=model.parameters(),lr=cfg['lr_shared']))
    opt=torch.optim.Adam(groups)
    store=PoolStore(out/'sampling_pools',cfg)
    visits=[0]*len(cases); order=[]; completed=0
    train_keys=prior['training_keys'] if prior else [list(k) for k in keys]
    effective=dict(records=manifest,config=cfg,budget_steps=steps,mode='latent_only' if prior else 'joint',
        training_keys=train_keys,pretrained_sha256=digest(pretrained) if pretrained else None,
        latent_optimizer='independent ParameterList; inactive grad=None, no inactive Adam updates',
        pool_refresh_unit='per-case visits',all_observed_planes_per_update=True)
    manifestpath=out/'fit_manifest.json'
    if resume and (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if ck['fit_contract']!=effective: raise ValueError('Resume configuration/source/observations changed')
        model.load_state_dict(ck['model'])
        with torch.no_grad():
            for code,value in zip(codes,ck['codes']): code.copy_(value)
        opt.load_state_dict(ck['optimizer']); visits=ck['visits']; order=ck['pending_order']; completed=ck['step']
        torch.set_rng_state(ck['torch_rng'].cpu()); np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng'])
        if str(device).startswith('cuda'): torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        # A crash can leave logged updates newer than the last atomic checkpoint.
        history=out/'history.jsonl'
        if history.exists():
            kept=[]
            for line in history.read_text().splitlines():
                try: row=json.loads(line)
                except json.JSONDecodeError: break
                if row['step']<=completed: kept.append(line)
            history.write_text(''.join(line+'\n' for line in kept))
    elif resume and any(out.iterdir()):
        # No checkpoint means training never committed an update. Only pool files
        # and the identical manifest may exist; refuse other partial artifacts.
        if manifestpath.exists() and json.loads(manifestpath.read_text())!=effective: raise ValueError('Initial resume mismatch')
    manifestpath.write_text(json.dumps(effective,indent=2)); (out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
    def save():
        payload=dict(format='plane_auto_occ_v1',model=model.state_dict(),codes=torch.stack([z.detach().cpu() for z in codes]),
            config=cfg,records=manifest,training_keys=train_keys,mode=effective['mode'],step=completed,
            optimizer=opt.state_dict(),visits=visits,pending_order=order,fit_contract=effective,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [])
        atomic_save(payload,out/'last.pt')
    if completed==0:
        save()
        if 0 in milestones and diagnose:
            nrng=np.random.get_state(); prng=random.getstate()
            devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
            try:
                with torch.random.fork_rng(devices=devices): diagnose(0,out/'last.pt',cases)
            finally: np.random.set_state(nrng); random.setstate(prng)
    with (out/'history.jsonl').open('a') as log:
        while completed<steps:
            if stop_requested and stop_requested(): save(); raise InterruptedError('Checkpoint saved for Slurm continuation')
            start=time.perf_counter()
            if not order: order=torch.randperm(len(cases)).tolist()
            idx=order.pop(); source=cases[idx]; sampler=store.get(source,visits[idx])
            c=dict(source)
            for k in ('s','axis','boundary','boundary_endo'):
                if k in c: c[k]=c[k].to(device)
            c['_area_sampler']=sampler
            opt.zero_grad(set_to_none=True)
            if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
            loss,stats=backward_case(model,codes[idx],c,cfg,completed)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite complete-case loss')
            params=[codes[idx]]+([] if prior else list(model.parameters()))
            norm=torch.nn.utils.clip_grad_norm_(params,cfg['clip_grad'],error_if_nonfinite=True)
            opt.step(); visits[idx]+=1; completed+=1
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            row=dict(step=completed,epoch=completed/len(cases),case_id=c['case_id'],phase=c['phase'],case_updates=visits[idx],
                loss=float(loss),gradient_norm=float(norm),pool_generation=sampler.meta['generation'],
                update_seconds=time.perf_counter()-start,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,**stats)
            log.write(json.dumps(row)+'\n'); log.flush()
            if completed==1 or completed%cfg['log_every']==0: print(json.dumps(row),flush=True)
            if completed%cfg['save_every']==0 or completed in milestones or (on_epoch and completed%len(cases)==0): save()
            if (completed in milestones and diagnose) or (completed%len(cases)==0 and on_epoch):
                # Evaluation may seed its fresh latent fits; preserve every training RNG.
                nrng=np.random.get_state(); prng=random.getstate()
                devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
                try:
                    with torch.random.fork_rng(devices=devices):
                        if completed in milestones and diagnose: diagnose(completed,out/'last.pt',cases)
                        if completed%len(cases)==0 and on_epoch: on_epoch(completed//len(cases),out/'last.pt',cases)
                finally: np.random.set_state(nrng); random.setstate(prng)
    save()
    # The shared model must be bit-identical during heldout-code optimization.
    if prior:
        for k,v in model.state_dict().items():
            if not torch.equal(v.cpu(),prior['model'][k].cpu()): raise AssertionError('Frozen shared weights changed')
    (out/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(step=completed,visits=visits,checkpoint_sha256=digest(out/'last.pt'),
        shared_frozen_verified=bool(prior),fit_contract=effective),indent=2))
    return out/'last.pt'
