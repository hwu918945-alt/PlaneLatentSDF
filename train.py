"""Reference single-process trainer; variable plane counts, patient-level batching."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from core.data import load_case, validate_split
from core.model import PlaneField, make_teacher, ema_update, leave_one_out_latent
from core.losses import DistanceWeightedDistillation
from core.losses import coordinate_gradient
from core.training import case_objective
from core.evaluation import case_geometry, aggregate_geometry
from core.initialization import initialize_decoder


def validate(student, paths, device):
    # Deterministic full observed sample encoding. No sampled midpoint labels.
    student.eval()
    scores = []
    with torch.no_grad():
        for path in paths:
            c = load_case(path, device)
            z = student.encoder(c['q'], c['sdf2d'], c['phase'])
            zl = leave_one_out_latent(c['s'], z)
            direct, loo, sign_accuracy = [], [], []
            for i, b in enumerate(c['boundary']):
                direct.append(student.decoder(b, z[i:i+1].expand(len(b), -1), c['phase']).abs().mean())
                mask = c['sdf2d'][i].abs() > 0.02
                if bool(mask.any()):
                    sq = c['q'][i, mask]
                    sf = student.decoder(sq, z[i:i+1].expand(len(sq), -1), c['phase'])
                    sd = c['sdf2d'][i, mask]
                    # Balanced accuracy avoids majority exterior labels hiding collapse.
                    groups = [(sf[sd < 0] < 0).float(), (sf[sd > 0] > 0).float()]
                    sign_accuracy.append(torch.stack([x.mean() for x in groups if len(x)]).mean())
                if 0 < i < len(z) - 1:
                    loo.append(student.decoder(b, zl[i-1:i].expand(len(b), -1), c['phase']).abs().mean())
            # |F| is only a distance surrogate while Eikonal is being learned.
            obs = float(torch.stack(direct).mean()) * c['scale_mm']
            hold = float(torch.stack(loo).mean()) * c['scale_mm'] if loo else obs
            sg = float(torch.stack(sign_accuracy).mean()) if sign_accuracy else 0.
            probes = 0.5*(c['boundary'][:-1, ::16] + c['boundary'][1:, ::16])
            with torch.enable_grad():
                qr = probes.reshape(-1, 3).detach().requires_grad_(True)
                values = student.field(qr, c['s'], z.detach(), c['axis'], c['phase'])
                grad = coordinate_gradient(values, qr, create_graph=False)
                eik = float(((grad.norm(dim=-1)-1)**2).mean())
            scores.append((obs, hold, sg, eik))
    student.train()
    obs, hold, sg, eik = np.asarray(scores).mean(0)
    return {'val_boundary_field_mm': float(obs), 'val_loo_boundary_field_mm': float(hold),
            'val_sign_balanced_accuracy': float(sg), 'val_eikonal': float(eik),
            'selection_score': float(obs + hold + 10*eik + 10*(1-sg))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.json')
    p.add_argument('--data', required=True, help='Root with train/ and val/ normalized NPZ files')
    p.add_argument('--output', default='runs/main')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs', type=int)
    p.add_argument('--max-steps', type=int, default=0, help='Smoke test only, 0=all steps')
    p.add_argument('--resume', default=None)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    torch.set_num_threads(cfg.get('cpu_threads', 1))
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    paths = {name: sorted((Path(args.data) / name).glob('*.npz')) for name in ['train', 'val', 'test']}
    if not paths['train'] or not paths['val']:
        raise ValueError('Need both train and val files')
    validate_split(paths)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / 'effective_config.json').write_text(json.dumps(cfg, indent=2))
    student = PlaneField(**cfg['model']).to(args.device)
    if cfg.get('decoder_initialization') not in [None,'geometric_sphere']:
        raise ValueError('Unknown decoder initialization')
    if cfg.get('decoder_initialization') == 'geometric_sphere':
        initialization = initialize_decoder(student)
        (out/'initialization.json').write_text(json.dumps(initialization,indent=2))
    teacher = make_teacher(student)
    distill = DistanceWeightedDistillation(prototypes=cfg['model']['prototypes']).to(args.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    start_epoch, global_step, best = 0, 0, math.inf
    if args.resume:
        # Only load a checkpoint you own. It includes optimizer state, not just weights.
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        if saved['config'] != cfg:
            raise ValueError('Resume config mismatch: restore exact original config')
        student.load_state_dict(saved['student']); teacher.load_state_dict(saved['teacher'])
        distill.load_state_dict(saved['distill']); optimizer.load_state_dict(saved['optimizer'])
        start_epoch, global_step, best = saved['epoch'] + 1, saved['global_step'], saved['best']
        random.setstate(saved['rng_python']); np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'].cpu())
        if torch.cuda.is_available() and saved.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['rng_cuda']])
    train_cases = [load_case(x, 'cpu') for x in paths['train']]
    for epoch in range(start_epoch, cfg['epochs']):
        epoch_start = time.monotonic()
        student.train(); teacher.eval()
        order = list(range(len(train_cases))); random.shuffle(order)
        records = []
        for start in range(0, len(order), cfg['batch_cases']):
            group = order[start:start + cfg['batch_cases']]
            optimizer.zero_grad(set_to_none=True)
            raw_logits = []
            for index in group:
                c = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in train_cases[index].items()}
                try:
                    loss, log, logits = case_objective(student, teacher, distill, c, cfg, epoch)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'Nonfinite loss: {log}')
                    (loss / len(group)).backward()
                except Exception as error:
                    (out/'failure_context.json').write_text(json.dumps(dict(epoch=epoch,global_step=global_step,case_id=c['case_id'],phase=c['phase'],error=str(error)),indent=2))
                    raise
                raw_logits.append(logits); records.append(log)
            try:
                grad = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg['clip_grad'], error_if_nonfinite=True)
            except Exception as error:
                (out/'failure_context.json').write_text(json.dumps(dict(epoch=epoch,global_step=global_step,cases=[train_cases[i]['path'] for i in group],error=str(error)),indent=2))
                raise
            optimizer.step()
            distill.update_center(torch.cat(raw_logits))
            progress = min(1., (epoch + 1) / cfg['epochs'])
            momentum = 1 - (1 - cfg['ema_start']) * (1 + math.cos(math.pi * progress)) / 2
            ema_update(teacher, student, momentum)
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break
        metrics = {k: float(np.mean([r[k] for r in records])) for k in records[0]}
        metrics.update(validate(student, paths['val'], args.device))
        metrics['proxy_selection_score'] = metrics.pop('selection_score')
        selection_epoch = epoch == 0 or (epoch+1)%cfg.get('geometry_validation_interval',5)==0 or epoch+1==cfg['epochs'] or bool(args.max_steps)
        if selection_epoch:
            student.eval()
            geometry = [case_geometry(student, load_case(path,args.device),cfg.get('validation_spacing_mm',1.0)) for path in paths['val']]
            geometric = aggregate_geometry(geometry)
            metrics.update({'val_'+k:v for k,v in geometric.items()})
            metrics['selection_score'] = geometric['direct_selection_penalty']+geometric['loo_selection_penalty']+10*metrics['val_eikonal']+10*(1-metrics['val_sign_balanced_accuracy'])
            student.train()
        metrics.update(epoch=epoch, global_step=global_step, grad_norm=float(grad), nonfinite_count=0, epoch_seconds=time.monotonic()-epoch_start)
        if not all(v is None or math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f'Nonfinite diagnostics at epoch {epoch}: {metrics}')
        improved = selection_epoch and metrics['selection_score'] < best
        if improved:
            best = metrics['selection_score']
        ckpt = dict(student=student.state_dict(), teacher=teacher.state_dict(), distill=distill.state_dict(),
                    optimizer=optimizer.state_dict(), epoch=epoch, global_step=global_step, best=best, config=cfg,
                    rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(ckpt, out / 'last.pt.tmp')
        (out / 'last.pt.tmp').replace(out / 'last.pt')
        if improved:
            torch.save(ckpt, out / 'best.pt.tmp')
            (out / 'best.pt.tmp').replace(out / 'best.pt')
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break
    if not args.max_steps:
        (out/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(epochs=cfg['epochs'],global_step=global_step,best_selection_score=best,checkpoint=str((out/'best.pt').resolve()),test_used_for_selection=False),indent=2))


if __name__ == '__main__':
    main()
