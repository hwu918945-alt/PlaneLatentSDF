"""Frozen-runtime Slurm deployment for full heart and 25-section solids."""
import argparse
import json
import shutil
import signal
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from .population import fit
from .protocol import evaluate
from .data import load_case
from .area_pool import digest

STOP=False

def stop_handler(signum,frame):
    global STOP
    STOP=True
    print(f'Signal {signum}: checkpoint and continue in a requeued job',flush=True)


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2))


def milestone(root,step,checkpoint,cases,plan,device):
    target=root/'outputs/fit'/f'step_{step:05d}'
    target.mkdir(parents=True,exist_ok=True)
    stable=target/'checkpoint.pt';shutil.copyfile(checkpoint,stable)
    for case in cases:
        evaluate(stable,case['path'],target/'diagnostics',device,plan['resolution'],plan['mesh_resolution'],observed_only=True)
    write(target/'VISUALIZATION_COMPLETE.json',dict(step=step,checkpoint_sha256=digest(stable),observed_only=True))


def verify_data(root,plan):
    lock=json.loads((root/'data_lock.json').read_text())
    for row in lock['source_copies']:
        # Audit local copies only; no access to the source repository is needed.
        if digest(row['local'])!=row['sha256']: raise ValueError(f"Local input changed: {row['local']}")
    if plan['kind']=='heart':
        for split in ('train','val','test'):
            for row in lock['splits'][split]['records']:
                p=root/row['path']
                if digest(p)!=row['sha256']: raise ValueError(f'Occupancy cache changed: {p}')
                case=load_case(p)
                if case['case_id']!=row['case_id'] or case['phase']!=row['phase']: raise ValueError('Case identity changed')
    elif digest(root/'data/case.npz')!=lock['case_sha256']: raise ValueError('Solid cache changed')
    return lock


def gate(root,plan,cfg,lock,device):
    marker=root/'gates/DEPLOYMENT_GATE_COMPLETE.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        if previous['config']!=cfg or previous['plan']!=plan: raise ValueError('Gate contract changed')
        return
    output=root/'gates'
    output.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-B','-m','unittest','plane_auto_occ.tests','plane_auto_occ.repair_tests',
        'plane_auto_occ.migration_tests','plane_auto_occ.population_tests'],check=True)
    if plan['kind']=='heart':
        paths=[str(root/r['path']) for r in lock['splits']['train']['records'][:2]];observed=[None,None]
    else: paths=[str(root/'data/case.npz')];observed=[None]
    config=dict(cfg);config['save_every']=1
    checkpoint=fit(paths,observed,config,output/'full_budget_gpu',device,steps=2,resume=True,stop_requested=lambda:STOP)
    history=[json.loads(x) for x in (output/'full_budget_gpu/history.jsonl').read_text().splitlines()]
    expected=[len(load_case(p)['s'])*cfg['points'] for p in paths]
    if len(history)!=2 or any(row['independent_points'] not in expected for row in history): raise ValueError('Full-budget gate failed')
    if not all(np.isfinite(row['loss']) and row['pool_fallbacks']==0 for row in history): raise ValueError('Invalid gate loss/sampling')
    if plan['kind']=='heart':
        p=str(root/lock['splits']['val']['records'][0]['path']);n=load_case(p)['total_planes']
        frozen=fit([p],[[0,n//2,n-1]],config,output/'frozen_gpu',device,steps=2,pretrained=checkpoint,
            resume=True,stop_requested=lambda:STOP)
        final=torch.load(frozen,map_location='cpu',weights_only=False);prior=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if not all(torch.equal(v,prior['model'][k]) for k,v in final['model'].items()):raise ValueError('Shared freeze failed')
    write(marker,dict(config=cfg,plan=plan,full_budget_updates=2,history=history,
        actual_device=device,inactive_latent_and_resume_tests=True,frozen_shared_verified=plan['kind']=='heart',
        scientific_success=False))


def heldout_split(root,split,shared,cfg,lock,plan,device,tag):
    base=root/'outputs'/tag/split;base.mkdir(parents=True,exist_ok=True);reports=[]
    for row in lock['splits'][split]['records']:
        path=root/row['path'];key=f"{row['case_id']}_{'ED' if row['phase']==0 else 'ES'}"
        caseout=base/key;caseout.mkdir(parents=True,exist_ok=True);n=row['planes']
        fitout=caseout/'fit'
        done=fitout/'TRAIN_COMPLETE.json'
        if done.exists():
            ck=torch.load(fitout/'last.pt',map_location='cpu',weights_only=False)
            if ck['step']!=plan['inference_steps'] or ck['fit_contract']['pretrained_sha256']!=digest(shared):
                raise ValueError('Heldout fit contract changed')
            fitted=fitout/'last.pt'
        else:
            fitted=fit([str(path)],[[0,n//2,n-1]],cfg,fitout,device,steps=plan['inference_steps'],
                pretrained=shared,resume=True,stop_requested=lambda:STOP)
        if STOP: raise InterruptedError('Continue heldout evaluation on requeue')
        metrics=caseout/'evaluation/metrics.json'
        report=json.loads(metrics.read_text()) if metrics.exists() else evaluate(fitted,path,caseout/'evaluation',device,
            plan['resolution'],plan['mesh_resolution'])
        if report['strict_heldout_mean_dice'] is None: raise ValueError('No independent heldout slice metric')
        reports.append(dict(case_id=row['case_id'],phase=row['phase'],strict_heldout_dice=report['strict_heldout_mean_dice'],
            observed_dice=report['observed_mean_dice'],metrics=str(metrics)))
        print(json.dumps(dict(split=split,index=len(reports),total=len(lock['splits'][split]['records']),**reports[-1])),flush=True)
    patient_ids=sorted(set(r['case_id'] for r in reports));patientmeans=[float(np.mean([r['strict_heldout_dice'] for r in reports if r['case_id']==p])) for p in patient_ids]
    summary=dict(split=split,shared_checkpoint=str(shared),shared_sha256=digest(shared),records=reports,
        patients=len(patient_ids),phase_records=len(reports),patient_mean_strict_heldout_dice=float(np.mean(patientmeans)),
        patient_std_strict_heldout_dice=float(np.std(patientmeans,ddof=1)),
        observation='first/middle/last only',inference_steps=plan['inference_steps'],shared_frozen=True)
    write(base/'SUMMARY.json',summary);return summary


def heart(root,plan,cfg,lock,device):
    paths=[str(root/r['path']) for r in lock['splits']['train']['records']]
    checkpoints=root/'outputs/checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
    selectionfile=root/'outputs/CHECKPOINT_SELECTION.json'
    def epoch_action(epoch,checkpoint,cases):
        if epoch not in plan['visual_epochs'] and epoch not in plan['val_epochs']:return
        stable=checkpoints/f'epoch_{epoch:04d}.pt'
        if not stable.exists():shutil.copyfile(checkpoint,stable)
        if epoch in plan['visual_epochs']:
            vis=root/'outputs/visualizations'/f'epoch_{epoch:04d}'
            if not (vis/'VISUALIZATION_COMPLETE.json').exists():
                for c in (cases[0],cases[len(cases)//2]):
                    evaluate(stable,c['path'],vis/f"{c['case_id']}_{c['phase']}",device,plan['resolution'],plan['mesh_resolution'],observed_only=True)
                write(vis/'VISUALIZATION_COMPLETE.json',dict(epoch=epoch,observed_only=True))
        if epoch in plan['val_epochs']:
            summarypath=root/'outputs'/f'validation_epoch_{epoch:04d}'/'val/SUMMARY.json'
            summary=json.loads(summarypath.read_text()) if summarypath.exists() else heldout_split(root,'val',stable,cfg,lock,plan,device,f'validation_epoch_{epoch:04d}')
            selection=json.loads(selectionfile.read_text()) if selectionfile.exists() else dict(candidates=[],selection_split='val',metric='patient_mean_strict_heldout_dice')
            if epoch not in [r['epoch'] for r in selection['candidates']]:
                selection['candidates'].append(dict(epoch=epoch,checkpoint=str(stable),sha256=digest(stable),score=summary['patient_mean_strict_heldout_dice']))
            best=max(selection['candidates'],key=lambda x:(x['score'],-x['epoch']))
            selection['best']=best;shutil.copyfile(best['checkpoint'],checkpoints/'best.pt');write(selectionfile,selection)
    out=root/'outputs/train'
    if (out/'last.pt').exists():
        current=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if current['step']%len(paths)==0:
            epoch_action(current['step']//len(paths),out/'last.pt',[load_case(p) for p in paths])
    shared=fit(paths,[None]*len(paths),cfg,out,device,steps=plan['total_updates'],resume=True,on_epoch=epoch_action,milestones=(0,),
        diagnose=lambda step,ck,cases:epoch_action(0,ck,cases),stop_requested=lambda:STOP)
    # Initial random field is generated independently so it cannot consume RNG.
    if not selectionfile.exists():raise ValueError('Validation never selected a checkpoint')
    best=checkpoints/'best.pt'
    test=root/'outputs/final_selected/test/SUMMARY.json'
    if not test.exists():heldout_split(root,'test',best,cfg,lock,plan,device,'final_selected')
    write(root/'outputs/RUN_COMPLETE.json',dict(train_checkpoint_sha256=digest(shared),selected=json.loads(selectionfile.read_text())['best'],
        test_summary=str(test),train_epochs=plan['train_epochs'],test_used_for_checkpoint_selection=False))


def solid(root,plan,cfg,device):
    case=root/'data/case.npz'
    checkpoint=fit([str(case)],[None],cfg,root/'outputs/fit',device,steps=plan['steps'],resume=True,
        milestones=plan['milestones'],diagnose=lambda step,ck,cases:milestone(root,step,ck,cases,plan,device),stop_requested=lambda:STOP)
    summary=root/'outputs/final_evaluation/metrics.json'
    if not summary.exists():evaluate(checkpoint,case,root/'outputs/final_evaluation',device,plan['resolution'],plan['mesh_resolution'],
        heldout_contours=root/'evaluation_only/heldout_contours.json',domain=root/'evaluation_only/evaluation_domain.json')
    write(root/'outputs/RUN_COMPLETE.json',dict(steps=plan['steps'],observed_planes=25,checkpoint_sha256=digest(checkpoint),
        final_metrics=str(summary),heldout_used_for_training_or_selection=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    root=Path(a.experiment).resolve();plan=json.loads((root/'plan.json').read_text());cfg=json.loads((root/'config.json').read_text())
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop_handler)
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    snapshot=json.loads((root/'runtime_lock.json').read_text())
    for row in snapshot['files']:
        if digest(root/row['path'])!=row['sha256']: raise ValueError(f"Frozen runtime/config changed: {row['path']}")
    torch.set_num_threads(cfg['threads']);lock=verify_data(root,plan)
    write(root/'logs/ENVIRONMENT.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name() if a.device.startswith('cuda') else 'cpu',plan=plan,config=cfg))
    try:
        gate(root,plan,cfg,lock,a.device)
        if plan['kind']=='heart':heart(root,plan,cfg,lock,a.device)
        else:solid(root,plan,cfg,a.device)
    except InterruptedError as error:
        print(str(error),flush=True);sys.exit(85)

if __name__=='__main__':main()
