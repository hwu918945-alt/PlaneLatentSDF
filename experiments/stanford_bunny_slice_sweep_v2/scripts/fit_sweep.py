"""One unchanged-method trajectory per slice count; fixed 6000-step EMA horizon."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from core.data import load_case
from core.model import PlaneField,make_teacher,ema_update
from core.losses import DistanceWeightedDistillation
from core.training import case_objective
from core.initialization import initialize_decoder


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--slices',type=int,required=True,choices=[16,25,50,75])
    parser.add_argument('--run-dir',default='runs/main')
    parser.add_argument('--stop-at',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();os.chdir(ROOT/f'slices_{args.slices:03d}')
    cfg=json.loads(Path('config.json').read_text())
    assert cfg['epochs']==6000
    limit=args.stop_at or cfg['epochs'];assert 0<limit<=cfg['epochs']
    torch.set_num_threads(1);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if args.device=='cuda':assert torch.cuda.is_available();torch.cuda.manual_seed_all(cfg['seed'])
    case=load_case('data/bunny_observed.npz',args.device);assert len(case['s'])==args.slices
    data_hash=hashlib.sha256(Path('data/bunny_observed.npz').read_bytes()).hexdigest()
    out=Path(args.run_dir);out.mkdir(parents=True,exist_ok=True)
    model=PlaneField(**cfg['model']).to(args.device);init=initialize_decoder(model)
    teacher=make_teacher(model);distill=DistanceWeightedDistillation(prototypes=cfg['model']['prototypes']).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    start=0;prior_elapsed=0.
    if args.resume and (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert saved['config']==cfg and saved['data_sha256']==data_hash
        model.load_state_dict(saved['student']);teacher.load_state_dict(saved['teacher'])
        distill.load_state_dict(saved['distill']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['global_step'];prior_elapsed=saved['elapsed_seconds']
        random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch'])
        if args.device=='cuda':torch.cuda.set_rng_state_all(saved['rng_cuda'])
        rows=(out/'metrics.jsonl').read_text().splitlines()
        rows=[r for r in rows if json.loads(r)['global_step']<=start]
        assert len(rows)==start
        (out/'metrics.jsonl').write_text('\n'.join(rows)+'\n')
        print(f'Restored optimizer, teacher, distillation center and RNG at step {start}',flush=True)
    elif (out/'metrics.jsonl').exists() or (out/'last.pt').exists():
        raise RuntimeError('Existing run: use --resume or a new run directory')
    (out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
    if start==0:(out/'initialization.json').write_text(json.dumps(init,indent=2))
    started=time.monotonic()
    for epoch in range(start,limit):
        optimizer.zero_grad(set_to_none=True)
        loss,logs,logits=case_objective(model,teacher,distill,case,cfg,epoch)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {epoch}: {logs}')
        loss.backward()
        assert all(p.grad is None for p in teacher.parameters())
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad'],error_if_nonfinite=True)
        optimizer.step();distill.update_center(logits)
        progress=(epoch+1)/cfg['epochs'];momentum=1-(1-cfg['ema_start'])*(1+math.cos(math.pi*progress))/2
        ema_update(teacher,model,momentum)
        elapsed=prior_elapsed+time.monotonic()-started
        logs.update(epoch=epoch,global_step=epoch+1,grad_norm=float(grad),elapsed_seconds=elapsed,nonfinite_count=0,observed_planes=args.slices)
        with (out/'metrics.jsonl').open('a') as handle:handle.write(json.dumps(logs,allow_nan=False)+'\n')
        if epoch==0 or (epoch+1)%100==0:print(json.dumps(logs),flush=True)
        if (epoch+1)%200==0 or epoch+1==limit:
            saved=dict(student=model.state_dict(),teacher=teacher.state_dict(),distill=distill.state_dict(),optimizer=optimizer.state_dict(),
                config=cfg,epoch=epoch,global_step=epoch+1,data_sha256=data_hash,elapsed_seconds=elapsed,
                rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all() if args.device=='cuda' else None)
            torch.save(saved,out/'last.pt.tmp');(out/'last.pt.tmp').replace(out/'last.pt')
            if epoch+1 in [2000,4000,6000]:
                target=out/f'step_{epoch+1:06d}.pt'
                if not target.exists():target.hardlink_to(out/'last.pt')
    if limit==cfg['epochs']:
        assert all((out/f'step_{step:06d}.pt').exists() for step in [2000,4000,6000])
        (out/'FITTING_COMPLETE.json').write_text(json.dumps(dict(optimizer_steps=6000,observed_planes=args.slices,
            milestones=[2000,4000,6000],ema_horizon=6000,heldout_used_for_loss_or_selection=False,reference_mesh_used_for_loss=False),indent=2))


if __name__=='__main__':main()
