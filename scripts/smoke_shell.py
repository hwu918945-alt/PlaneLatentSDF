"""Fresh-weight optimizer check with all losses; never resumes the research run."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core.data import load_case,require_shell
from core.model import PlaneField,make_teacher,ema_update
from core.losses import RealPlaneDistillation
from core.training import case_objective
from core.sampling import validate_training_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path);p.add_argument('--device',default='cpu')
    p.add_argument('--config',default='config.json');p.add_argument('--output',default='preflight/shell_smoke.json')
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(11)
    cfg=json.loads(Path(args.config).read_text());validate_training_config(cfg)
    paths=sorted((args.data/'train').glob('*.npz'))
    if not paths:raise ValueError('Smoke optimizer uses only the train split')
    cases=[load_case(path,args.device) for path in paths[:2]]
    for c in cases:require_shell(c)
    model=PlaneField(**cfg['model']).to(args.device);teacher=make_teacher(model)
    distill=RealPlaneDistillation(prototypes=cfg['model']['prototypes']).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr']);logs=[]
    for epoch in [0,cfg['warmup_epochs']+cfg['ramp_epochs']]:
        optimizer.zero_grad(set_to_none=True);raw=[]
        for c in cases:
            loss,log,logits=case_objective(model,teacher,distill,c,cfg,epoch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            (loss/len(cases)).backward();raw.append(logits);logs.append(log)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad'],error_if_nonfinite=True)
        optimizer.step();distill.update_center(torch.cat(raw));ema_update(teacher,model,cfg['ema_start'])
        if any(p.grad is not None for p in teacher.parameters()):raise AssertionError('Teacher received gradients')
        if logs[-1]['weighted_eikonal']<=0:raise AssertionError('Eikonal disabled')
    report=dict(status='PASS',device=args.device,torch=torch.__version__,grad_norm=float(norm),losses=logs,
                used_split='train',weights_discarded=True,cardiac_quality_verified=False)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
