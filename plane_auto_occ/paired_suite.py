"""Frozen single-case-from-zero A/B/C/D/FULL suite for the known worst ten."""
import argparse,json,shutil,sys,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from .area_pool import digest
from .data import load_case
from .paired_pools import build

PATIENTS=['008','055','159','206','021','358','029','343','352','138']
ARMS=['A','B','C','D','FULL']


def arm_config(base,arm,root,refs):
    if arm not in ARMS:raise ValueError('Unknown arm')
    cfg=dict(base)
    cfg.update(pipeline_backend='paired_roi_exact_v1',sampling_version='paired_multiscale_roi_v1',
        objective_version='paired_bce_all_v1' if arm=='A' else 'paired_bce_roi_dice_v1',
        dice_weight=0. if arm=='A' else 1.,dice_eps=1e-6,
        decoder_final_activation='softplus' if arm in ('A','B') else 'leaky_relu',
        decoder_residual='h2_h4' if arm in ('D','FULL') else 'none',decoder_leaky_slope=.01,
        pair_scales_mm=[.25,.75,2.],pair_scale_weights=[.5,.3,.2],
        roi_reference_observed=refs,offline_pool_root=str(Path(root).resolve()/'point_pools'),
        collapse_std_threshold=1e-3)
    # Preserve all optimizer, update and architecture settings from old per-phase fit.
    if cfg['seam_weight']!=0:raise ValueError('Expected original Seam=0')
    return cfg


def _previous_record(previous,patient,phase):
    key=f"{patient}_{'ED' if phase==0 else 'ES'}";fit=Path(previous)/'outputs/single_case'/key/'fit'
    manifest=fit/'fit_manifest.json'
    if manifest.exists():
        obj=json.loads(manifest.read_text());rows=obj['records'];cfg=obj['config']
    elif (fit/'last.pt').exists():
        import torch
        ck=torch.load(fit/'last.pt',map_location='cpu',weights_only=False);rows=ck['records'];cfg=ck['config']
    else:raise FileNotFoundError(f'Original single-case metadata not found: {fit}; supply --records-json')
    if len(rows)!=1:raise ValueError(f'Expected original single-case fit: {fit}')
    row=dict(rows[0]);row['config']=cfg;return row


def prepare(previous,output,records_json=None,steps=None):
    root=Path(output).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError('Use a NEW suite directory')
    if records_json:
        supplied=json.loads(Path(records_json).read_text())
        rows=supplied['records'];base_cfg=supplied.get('config')
        if len(rows)!=20:raise ValueError('Exactly twenty ED/ES records required')
    else:
        rows=[_previous_record(previous,p,phase) for p in PATIENTS for phase in (0,1)];base_cfg=None
    root.mkdir(parents=True,exist_ok=True);(root/'data').mkdir();copied=[];keys=set();global_settings=None
    frozen_keys=['latent_dim','width','points','steps','seed','lr_shared','lr_latent','clip_grad','latent_weight',
                 'loo_mix','loo_weight','direct_weight','data_weight','seam_weight','pool_refresh_updates','pool_margin_mm','boundary_band_mm']
    for row in rows:
        path=Path(row['path']).expanduser().resolve();c=load_case(path)
        patient=str(c['case_id']).zfill(3);phase=int(c['phase']);key=(patient,phase)
        if patient not in PATIENTS or key in keys:raise ValueError(f'Unexpected/duplicate case phase {key}')
        keys.add(key)
        if 'source_sha256' in row and digest(path)!=row['source_sha256']:raise ValueError('Original cache source changed')
        cfg=dict(row.get('config') or base_cfg or {})
        if not cfg:raise ValueError('Need original fit config, not guessed optimizer settings')
        # Old fit() may override budget_steps without updating cfg['steps']; the
        # confirmed worst-ten protocol is 2000 updates, not the shared-training budget.
        cfg['steps']=int(steps) if steps is not None else 2000
        if cfg['points']!=32768 and steps is None:raise ValueError('Original experiment expected 32768 points per plane')
        if len(c['s'])<3:raise ValueError('Original full-SAX fit needs at least three planes')
        settings={k:cfg[k] for k in frozen_keys}
        if global_settings is None:global_settings=settings
        elif settings!=global_settings:raise ValueError('Original twenty phase configurations differ; resolve before paired comparisons')
        dst=root/'data'/f"{patient}_{'ED' if phase==0 else 'ES'}.npz";shutil.copy2(path,dst)
        refs=row.get('roi_reference_observed',[0,len(c['s'])//2,len(c['s'])-1])
        if len(refs)!=3 or refs!=sorted(set(refs)):raise ValueError('Expected original first/middle/last ROI reference')
        copied.append(dict(patient=patient,case_id=c['case_id'],phase=phase,path=str(dst.relative_to(root)),
            original_path=str(path),source_sha256=digest(dst),planes=len(c['s']),observed=list(range(len(c['s']))),
            roi_reference_observed=refs,base_config=cfg))
    if keys!={(p,phase) for p in PATIENTS for phase in (0,1)}:raise ValueError('Incomplete ten-patient ED/ES set')
    package=Path(__file__).resolve().parent
    shutil.copytree(package,root/'runtime/plane_auto_occ',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    tasks=[]
    for arm in ARMS:
        for record in copied:
            key=f"{record['patient']}_{'ED' if record['phase']==0 else 'ES'}";out=root/'outputs'/arm/key
            config=root/'configs'/arm/f'{key}.json';config.parent.mkdir(parents=True,exist_ok=True)
            cfg=arm_config(record['base_config'],arm,root,record['roi_reference_observed'])
            config.write_text(json.dumps(cfg,indent=2))
            tasks.append(dict(task_index=len(tasks),arm=arm,key=key,record=record['path'],config=str(config.relative_to(root)),
                output=str(out.relative_to(root))))
    plan=dict(schema='paired_worst10_suite_v1',patients=PATIENTS,phase_records=20,tasks=tasks,records=copied,
        arms=ARMS,full_equals_D=True,full_role='independent from-zero repeat of D, same seed and sampling',
        original_previous_experiment=str(Path(previous).resolve()) if previous else None,
        initialization='fresh G,D,z per patient/phase/arm; no pretrained weights or old Adam',
        protocol=global_settings,all_cached_sax_supervised=True,original_hidden_subset_now_fit=True,
        resolution=160,mesh_resolution=96,threshold_logit=0.,roi='original first/middle/last bbox + 20mm')
    (root/'plan.json').write_text(json.dumps(plan,indent=2))
    files=list((root/'runtime').rglob('*.py'))+list((root/'configs').rglob('*.json'))+[root/'plan.json']
    lock={str(p.relative_to(root)):digest(p) for p in files}
    (root/'runtime_lock.json').write_text(json.dumps(lock,indent=2))
    print(json.dumps(dict(suite=str(root),patients=10,phase_records=20,runs=len(tasks),steps=global_settings['steps'],full_equals_D=True),indent=2))
    return root


def _plan(root):
    root=Path(root).resolve();return root,json.loads((root/'plan.json').read_text())


def verify(root):
    root=Path(root)
    for name,sha in json.loads((root/'runtime_lock.json').read_text()).items():
        if digest(root/name)!=sha:raise ValueError(f'Frozen runtime/config changed: {name}')


def precache(root,workers=3,generations=None):
    root,plan=_plan(root);verify(root);tasks=[]
    for record in plan['records']:
        case=root/record['path'];key=f"{record['patient']}_{'ED' if record['phase']==0 else 'ES'}"
        cfg=json.loads((root/'configs/A'/f'{key}.json').read_text())
        if digest(case)!=record['source_sha256']:raise ValueError('Cache bytes changed')
        gens=range((cfg['steps']+79)//80) if generations is None else generations
        tasks.extend((case,record['observed'],cfg,int(g)) for g in gens)
    complete=[];failed=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending={executor.submit(build,p,obs,root/'point_pools',cfg,g):(p,g) for p,obs,cfg,g in tasks}
        for future in as_completed(pending):
            p,g=pending[future]
            try: m=future.result();complete.append(dict(case=p.name,generation=g))
            except Exception as e:failed.append(dict(case=p.name,generation=g,error=repr(e)))
            if (len(complete)+len(failed))%20==0:print(json.dumps(dict(precache_completed=len(complete),failed=len(failed),total=len(tasks))),flush=True)
    (root/'PRECACHE_STATUS.json').write_text(json.dumps(dict(complete=complete,failed=failed,total=len(tasks)),indent=2))
    if failed:raise RuntimeError(f'{len(failed)} pools failed; all failed phases retained in PRECACHE_STATUS.json')


def run_task(root,index,device='cuda',resume=False):
    from .population import fit
    from .paired_evaluate import evaluate
    root,plan=_plan(root);verify(root);task=plan['tasks'][int(index)]
    cfg=json.loads((root/task['config']).read_text());case=root/task['record'];out=root/task['output']
    out.mkdir(parents=True,exist_ok=True)
    done=out/'RUN_COMPLETE.json'
    if done.exists():
        if resume:return json.loads(done.read_text())
        raise FileExistsError(f'Already completed {out}; default means fresh, not skip/resume')
    try:
        checkpoint=fit([str(case)],[None],cfg,out/'fit',device,steps=cfg['steps'],pretrained=None,resume=resume)
        report=evaluate(checkpoint,case,out/'evaluation',device,plan['resolution'],plan['mesh_resolution'])
        result=dict(task_index=int(index),arm=task['arm'],key=task['key'],steps=cfg['steps'],
            checkpoint_sha256=digest(checkpoint),all_sax_mean_dice=report['all_sax_mean_dice'],
            original_hidden_subset_fit_mean_dice=report['original_hidden_subset_fit_mean_dice'],
            mesh_status=report['mesh'].get('status','ZERO_CROSSING'),initialization='fresh unless explicitly resumed this new fit')
        done.write_text(json.dumps(result,indent=2));(out/'RUN_FAILED.json').unlink(missing_ok=True);return result
    except Exception as e:
        (out/'RUN_FAILED.json').write_text(json.dumps(dict(task=task,error=repr(e),traceback=traceback.format_exc()),indent=2));raise


def summarize(root):
    root,plan=_plan(root);results={};table=[]
    for arm in ARMS:
        rows=[];failures=[];pending=[]
        for task in [t for t in plan['tasks'] if t['arm']==arm]:
            out=root/task['output'];m=out/'evaluation/metrics.json'
            if (out/'RUN_COMPLETE.json').exists() and m.exists():
                r=json.loads(m.read_text());rows.append(dict(key=task['key'],**r))
            elif (out/'RUN_FAILED.json').exists():failures.append(task['key'])
            else:pending.append(task['key'])
        means={}
        for patient in PATIENTS:
            rr=[r for r in rows if r['key'].split('_')[0]==patient]
            means[patient]=dict(all_sax=float(np.mean([r['all_sax_mean_dice'] for r in rr])) if len(rr)==2 else None,
                original_hidden_now_fit=float(np.mean([r['original_hidden_subset_fit_mean_dice'] for r in rr])) if len(rr)==2 else None)
        complete=len(rows)==20
        result=dict(completed=len(rows),failed=failures,pending=pending,patients=means,
            patient_mean_all_sax=float(np.mean([v['all_sax'] for v in means.values()])) if complete else None,
            patient_mean_original_hidden_now_fit=float(np.mean([v['original_hidden_now_fit'] for v in means.values()])) if complete else None,
            zero_crossing_phases=sum(r['mesh'].get('status')!='NO_ZERO_CROSSING' for r in rows),
            empty_prediction_planes=sum(p['empty_prediction'] for r in rows for p in r['fixed_grid_planes']),
            independent_generalization_claim=False)
        results[arm]=result;table.append(f"| {arm} | {len(rows)}/20 | {result['patient_mean_all_sax']} | {result['patient_mean_original_hidden_now_fit']} | {result['zero_crossing_phases']} |")
    summary=dict(arms=results,full_equals_D=True,full_is_repeat_not_new_ablation=True,failures_never_dropped=True)
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    (root/'SUMMARY.md').write_text('# Paired worst-ten suite\n\n原隐藏切片已参与全SAX监督，成绩为拟合成绩。FULL与D配置相同，是复跑。\n\n|组|完成阶段|患者均值 All SAX|原隐藏子集拟合均值|有零面阶段|\n|---|---:|---:|---:|---:|\n'+'\n'.join(table)+'\n')
    print(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--previous-experiment');a.add_argument('--records-json');a.add_argument('--output',required=True);a.add_argument('--steps',type=int)
    a=sub.add_parser('precache');a.add_argument('--suite',required=True);a.add_argument('--workers',type=int,default=3);a.add_argument('--generations',type=int,nargs='+')
    a=sub.add_parser('run');a.add_argument('--suite',required=True);a.add_argument('--task-index',type=int);a.add_argument('--arm',choices=ARMS);a.add_argument('--device',default='cuda');a.add_argument('--resume',action='store_true')
    a=sub.add_parser('summarize');a.add_argument('--suite',required=True)
    args=p.parse_args()
    if args.command=='prepare':
        if not args.previous_experiment and not args.records_json:p.error('Need --previous-experiment or --records-json')
        prepare(args.previous_experiment,args.output,args.records_json,args.steps)
    elif args.command=='precache':precache(args.suite,args.workers,args.generations)
    elif args.command=='summarize':summarize(args.suite)
    else:
        root,plan=_plan(args.suite)
        if args.task_index is not None:run_task(root,args.task_index,args.device,args.resume)
        else:
            failed=[]
            for task in plan['tasks']:
                if args.arm and task['arm']!=args.arm:continue
                try:run_task(root,task['task_index'],args.device,args.resume)
                except Exception as e:failed.append((task['task_index'],repr(e)));print(f'FAILED {task["key"]} {task["arm"]}: {e}',flush=True)
            summarize(root)
            if failed:raise SystemExit(1)


if __name__=='__main__':main()
