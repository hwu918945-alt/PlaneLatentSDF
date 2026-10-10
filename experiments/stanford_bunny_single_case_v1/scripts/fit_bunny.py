"""Fixed-step single-case parameter fitting using an unchanged method snapshot."""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from core.data import load_case
from core.model import PlaneField,make_teacher,ema_update
from core.losses import DistanceWeightedDistillation
from core.training import case_objective
from core.initialization import initialize_decoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int);args=p.parse_args()
    cfg=json.loads(Path('config.json').read_text())
    if args.steps:cfg['epochs']=args.steps
    torch.set_num_threads(1);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if args.device=='cuda':
        assert torch.cuda.is_available();torch.cuda.manual_seed_all(cfg['seed'])
    out=Path('runs/smoke' if args.steps else 'runs/main');out.mkdir(parents=True,exist_ok=True)
    (out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
    case=load_case('data/bunny_observed.npz',args.device)
    model=PlaneField(**cfg['model']).to(args.device);init=initialize_decoder(model)
    teacher=make_teacher(model);distill=DistanceWeightedDistillation(prototypes=cfg['model']['prototypes']).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    (out/'initialization.json').write_text(json.dumps(init,indent=2))
    started=time.monotonic()
    for epoch in range(cfg['epochs']):
        optimizer.zero_grad(set_to_none=True)
        loss,logs,logits=case_objective(model,teacher,distill,case,cfg,epoch)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {epoch}: {logs}')
        loss.backward()
        if any(p.grad is not None for p in teacher.parameters()):raise AssertionError('Teacher received gradients')
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad'],error_if_nonfinite=True)
        optimizer.step();distill.update_center(logits)
        progress=(epoch+1)/cfg['epochs'];momentum=1-(1-cfg['ema_start'])*(1+math.cos(math.pi*progress))/2
        ema_update(teacher,model,momentum)
        logs.update(epoch=epoch,global_step=epoch+1,grad_norm=float(grad),elapsed_seconds=time.monotonic()-started,nonfinite_count=0)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(logs,allow_nan=False)+'\n')
        if epoch==0 or (epoch+1)%50==0:print(json.dumps(logs),flush=True)
        if (epoch+1)%200==0 or epoch+1==cfg['epochs']:
            saved=dict(student=model.state_dict(),teacher=teacher.state_dict(),distill=distill.state_dict(),optimizer=optimizer.state_dict(),
                       config=cfg,epoch=epoch,global_step=epoch+1,rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                       rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if args.device=='cuda' else None)
            torch.save(saved,out/'last.pt.tmp');(out/'last.pt.tmp').replace(out/'last.pt')
    (out/'FITTING_COMPLETE.json').write_text(json.dumps(dict(case_id='stanford_bunny',optimizer_steps=cfg['epochs'],
                      reference_mesh_used_for_loss=False,heldout_used_for_loss_or_selection=False,checkpoint_selection='fixed final step',
                      clinical_or_generalization_quality_claimed=False),indent=2))


if __name__=='__main__':main()
