"""Build deterministic pools before each GPU-training stage, with resumable files."""
import argparse,json,time,signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import torch
from .offline_pools import prepare,directory,validate
from .data import load_case
from .area_pool import digest

STOP=False
def stop_handler(signum,frame):
    global STOP
    STOP=True
    print('Precache signal: finish active cases; completed atomic pools survive requeue',flush=True)


def stage(root,split,generations=None,start=0,end=None,workers=4,backend='compiled_cpu'):
    root=Path(root);cfg=json.loads((root/'config.json').read_text());plan=json.loads((root/'plan.json').read_text())
    lock=json.loads((root/'data_lock.json').read_text());records=lock['splits'][split]['records'][start:end]
    total=plan['train_epochs'] if split=='train' else plan['inference_steps']
    generations=list(range((total+cfg['pool_refresh_updates']-1)//cfg['pool_refresh_updates'])) if generations is None else list(generations)
    cache=Path(cfg['offline_pool_root']);legacy=cfg.get('legacy_pool_root') if split=='train' else None
    torch.set_num_threads(1)
    # Compile once before worker threads enter the classifier.
    if backend=='compiled_cpu':
        import numpy as np
        from .fast_geometry import CompiledPlaneClassifier
        CompiledPlaneClassifier(np.array([[-2.,-2.],[2.,-2.],[2.,2.],[-2.,2.]]),np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])).classify(np.zeros((1,2)),.05,3.)
    begun=time.perf_counter();tasks=[]
    for row in records:
        path=root/row['path']
        if digest(path)!=row['sha256']:raise ValueError('Input changed')
        observed=None if split=='train' else [0,row['planes']//2,row['planes']-1]
        tasks.extend((path,observed,g) for g in generations)
    complete=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending={executor.submit(prepare,p,o,cache,cfg,g,legacy,backend):(p,o,g) for p,o,g in tasks}
        for future in as_completed(pending):
            if STOP:
                for queued in pending:queued.cancel()
                raise InterruptedError('Precache interrupted; resume existing verified pools')
            m=future.result();p,o,g=pending[future];c=load_case(p,observed=o)
            complete.append(dict(case_id=m['case_id'],phase=m['phase'],generation=g,path=str(directory(cache,c,g)),pool_sha256=m['pool_sha256'],backend=m['backend'],duplicates_replaced=m['duplicates_replaced']))
            if len(complete)%20==0 or len(complete)==len(tasks):print(json.dumps(dict(split=split,complete=len(complete),total=len(tasks),elapsed_seconds=time.perf_counter()-begun)),flush=True)
    marker=root/'cache_manifests'/f'{split}_g{"-".join(map(str,generations))}_{start}_{end or len(lock["splits"][split]["records"])}.json'
    marker.parent.mkdir(exist_ok=True);marker.write_text(json.dumps(dict(schema='offline_stage_ready_v1',split=split,records=complete,elapsed_seconds=time.perf_counter()-begun,config_sha256=digest(root/'config.json'),hidden_geometry_used=False),indent=2))
    return marker


def ensure(root,split):
    """Barrier outside latent optimization; cached complete stages only read manifests."""
    root=Path(root);cfg=json.loads((root/'config.json').read_text());lock=json.loads((root/'data_lock.json').read_text())
    records=lock['splits'][split]['records'];plan=json.loads((root/'plan.json').read_text())
    total=plan['train_epochs'] if split=='train' else plan['inference_steps']
    generations=range((total+cfg['pool_refresh_updates']-1)//cfg['pool_refresh_updates'])
    missing=False
    for row in records:
        c=load_case(root/row['path'],observed=None if split=='train' else [0,row['planes']//2,row['planes']-1])
        for g in generations:
            if not directory(cfg['offline_pool_root'],c,g).exists():missing=True;break
        if missing:break
    if missing:
        print(f'OFFLINE_STAGE_BEGIN {split}: optimization waits until all pools are ready',flush=True)
        stage(root,split,workers=cfg.get('cache_workers',3))
        print(f'OFFLINE_STAGE_READY {split}',flush=True)


def prune(root,split,generation):
    """Delete only new-run arrays after all cases have consumed that generation."""
    root=Path(root);cfg=json.loads((root/'config.json').read_text());cache=Path(cfg['offline_pool_root']).resolve()
    if cache!=root.resolve()/'point_pools':raise ValueError('Refuse pruning outside this experiment')
    lock=json.loads((root/'data_lock.json').read_text());removed=[]
    for row in lock['splits'][split]['records']:
        c=load_case(root/row['path'],observed=None if split=='train' else [0,row['planes']//2,row['planes']-1])
        p=directory(cache,c,generation)
        if not p.exists():continue
        m=json.loads((p/'metadata.json').read_text())
        for name in m['files']:
            if (p/name).exists():(p/name).unlink()
        (p/'metadata.json').replace(p.with_suffix('.retired.json'))
        p.rmdir()
        removed.append(str(p))
    log=root/'cache_manifests'/f'PRUNED_{split}_{generation:04d}.json';log.parent.mkdir(exist_ok=True)
    log.write_text(json.dumps(dict(paths=removed,metadata_retained=True,scope='new experiment arrays only'),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--split',choices=['train','val','test'],required=True)
    p.add_argument('--generations',type=int,nargs='+');p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--backend',default='compiled_cpu');a=p.parse_args()
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop_handler)
    try:stage(a.experiment,a.split,a.generations,a.start,a.end,a.workers,a.backend)
    except InterruptedError as error:
        print(str(error),flush=True);raise SystemExit(85)
if __name__=='__main__':main()
