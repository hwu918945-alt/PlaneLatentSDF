"""Frozen A/B/C/D/FULL solid benchmarks; one GPU worker per arm, two cases serially."""
import argparse
import fcntl
import json
import shutil
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .area_pool import digest
from .data import load_case
from .inference import bounds,restore
from .model import model_from_config
from .paired_pools import build,location,check,PairedSampler,_classifier
from .paired_suite import ARMS,arm_config
from .population import fit
from .protocol import evaluate


KINDS=['bunny','balloon_dog']


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def fingerprints(checkpoint):
    import hashlib
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    result={}
    for prefix in ['generator.','decoder.','codes']:
        h=hashlib.sha256();state={'codes':ck['codes']} if prefix=='codes' else {k:v for k,v in ck['model'].items() if k.startswith(prefix)}
        if not state:raise ValueError(f'Unknown model prefix {prefix}')
        for name,tensor in sorted(state.items()):
            a=tensor.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(a.tobytes())
        result[prefix]=h.hexdigest()
    return result


def prepare(repository,root):
    repository=Path(repository).resolve();root=Path(root).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError('New suite directory required')
    root.mkdir(parents=True,exist_ok=True);records=[];tasks=[]
    for kind in KINDS:
        old=repository/'runs'/f'{kind}_25_20261011_v1';source=old/'data/case.npz';c=load_case(source)
        if c.get('schema')!='plane_auto_occ_solid_v1' or c['total_planes']!=25:raise ValueError('Expected original 25 observed solid sections')
        data=root/'data'/kind;data.mkdir(parents=True)
        for name in ['case.npz','observed_contours.json']:shutil.copy2(old/'data'/name,data/name)
        evaluation=root/'evaluation_only'/kind;evaluation.mkdir(parents=True)
        for name in ['heldout_contours.json','evaluation_domain.json','reference.ply']:
            shutil.copy2(old/'evaluation_only'/name,evaluation/name)
        old_cfg=read(old/'config.json');old_plan=read(old/'plan.json')
        assert old_cfg['steps']==4000 and old_plan['steps']==4000
        records.append(dict(kind=kind,case_id=c['case_id'],path=str((data/'case.npz').relative_to(root)),
                            source=str(source),source_sha256=digest(source),observed=list(range(25)),planes=25,
                            baseline_experiment=str(old),base_config=old_cfg))
    for arm in ARMS:
        for record in records:
            cfg=arm_config(record['base_config'],arm,root,record['observed'])
            cfg.update(sampling_case_geometry='solid_even_odd_pairs_v1',fast_query_points=524288,
                       prefetch_cases=1,sampler_cache_cases=1,checkpoint_seconds=300)
            config=root/'configs'/arm/f"{record['kind']}.json";write(config,cfg)
            tasks.append(dict(arm=arm,kind=record['kind'],case=record['path'],config=str(config.relative_to(root)),
                              output=f"outputs/{arm}/{record['kind']}"))
    plan=dict(kind='paired_solids25_v1',arms=ARMS,kinds=KINDS,records=records,tasks=tasks,
              observed_planes=25,steps=4000,milestones=[0,1000,2000,3000,4000],resolution=160,mesh_resolution=128,
              initialization='independent random G/D/z/Adam, seed19; no pretrained weights',
              sampling='all real even/odd boundaries; two exchangeable boundary pair banks; uniform ROI',
              training_ROI='bbox of every observed real ring +20mm',
              final_domain='original frozen evaluation-only domain; strict heldout rings read only after step4000',
              full_is_repeat_of_D=True,expected_models=10,expected_pools=100)
    write(root/'plan.json',plan)
    shutil.copytree(Path(__file__).resolve().parent,root/'runtime/plane_auto_occ',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files=list((root/'runtime').rglob('*.py'))+list((root/'configs').rglob('*.json'))+list((root/'data').rglob('*'))+list((root/'evaluation_only').rglob('*'))+[root/'plan.json']
    write(root/'runtime_lock.json',{str(p.relative_to(root)):digest(p) for p in files if p.is_file()})
    print(json.dumps(dict(suite=str(root),models=10,pools=100,steps=4000),indent=2),flush=True)


def verify(root):
    root=Path(root).resolve()
    for name,sha in read(root/'runtime_lock.json').items():
        if digest(root/name)!=sha:raise ValueError(f'Frozen source/config/input changed: {name}')
    return root,read(root/'plan.json')


def pool_audit(root,record,cfg,generation,all_points=False):
    c=load_case(root/record['path']);path=location(root/'point_pools',c,cfg,generation)
    m=check(path,c,cfg,generation,full=True)
    if all_points:
        lower,upper,basis=bounds(c,20.)
        q=np.load(path/'q.npy',mmap_mode='r');labels=np.load(path/'region.npy',mmap_mode='r');source=np.load(path/'source.npy',mmap_mode='r')
        if q.shape!=(25,cfg['points']*8,3):raise ValueError('Changed full point-pool capacity')
        for i,pos in enumerate(c['s'].numpy()):
            plane=_classifier(c,cfg,i,basis,lower,upper)
            actual,valid,d=plane.classify(q[i].astype(np.float64)@basis[:,:2]*c['scale_mm'])
            assert valid.all() and d.min()>.05 and np.array_equal(actual,labels[i])
            assert len(np.unique(q[i],axis=0))==len(q[i])
            assert [(source[i]==k).sum() for k in range(5)]==[24576]*4+[163840]
            assert np.array_equal(actual[source[i]<4],np.array([0,1,0,1])[source[i][source[i]<4]])
    return dict(kind=record['kind'],generation=generation,path=str(path),metadata_sha256=digest(path/'metadata.json'))


def gate(root,device):
    root,plan=verify(root)
    if not str(device).startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Full-budget gate requires CUDA')
    rows=[];fp={}
    for arm in ARMS:
        for record in plan['records']:
            kind=record['kind'];cfg=read(root/f'configs/{arm}/{kind}.json');case=root/record['path']
            out=root/'gates/full_budget'/arm/kind
            checkpoint=fit([str(case)],[None],cfg,out/'fit',device,steps=2,pretrained=None,resume=False)
            history=[json.loads(s) for s in (out/'fit/history.jsonl').read_text().splitlines()]
            assert len(history)==2 and all(r['independent_points']==819200 and np.isfinite(r['loss']) and r['pool_fallbacks']==0 for r in history)
            for r in history:
                assert all(p['source_counts']=={'0':3072,'1':3072,'2':3072,'3':3072,'4':20480} for p in r['sampling'])
            fp[(arm,kind)]=fingerprints(checkpoint)
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            torch.manual_seed(cfg['seed']);initial=model_from_config(cfg)
            initial_code=torch.randn(cfg['latent_dim'])*.01
            assert not torch.equal(ck['codes'][0],initial_code)
            for prefix in ['generator.','decoder.']:
                assert any(not torch.equal(v,initial.state_dict()[k]) for k,v in ck['model'].items() if k.startswith(prefix))
            model,code,u,obs,full=restore(checkpoint,case,device)
            with torch.no_grad():
                sample=full['q'][0,:7].to(device)
                original=model_from_config(cfg).to(device);original.load_state_dict(ck['model'])
                original_code=ck['codes'][0].to(device);original_u=original.plane_latents(original_code,obs['s'],obs['phase'])
                assert torch.equal(model.field(sample,obs['s'],u,obs['axis'],obs['phase']),original.field(sample,obs['s'],original_u,obs['axis'],obs['phase']))
            rows.append(dict(arm=arm,kind=kind,updates=2,points_per_update=819200,
                             generator_decoder_and_code_updated=True,restore_field_equal=True,
                             fingerprints=fp[(arm,kind)],history=history))
            if arm=='D':evaluate(checkpoint,case,out/'observed_interface_evaluation',device,32,24,observed_only=True)
            print(f'CUDA gate {arm}/{kind} passed',flush=True)
    assert all(fp[('D',kind)]==fp[('FULL',kind)] for kind in KINDS)
    write(root/'gates/GPU_GATE_COMPLETE.json',dict(device=device,cuda_name=torch.cuda.get_device_name(),
        full_budget_checks=rows,D_FULL_final_fingerprints_equal=True,scientific_training_complete=False))


def precache(root,device='cuda',workers=2):
    root,plan=verify(root)
    # Both real first generations and all five full-budget CUDA interfaces pass
    # before spending storage on the remaining offline generations.
    pilot=[]
    for record in plan['records']:
        cfg=read(root/f"configs/A/{record['kind']}.json")
        build(root/record['path'],None,root/'point_pools',cfg,0)
        pilot.append(pool_audit(root,record,cfg,0,all_points=True))
    write(root/'gates/REAL_FIRST_GENERATION_COMPLETE.json',dict(records=pilot,all_final_points_reclassified=True))
    gate(root,device)
    jobs=[(record,g) for record in plan['records'] for g in range(1,50)]
    finished=list(pilot);failed=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map={executor.submit(build,root/r['path'],None,root/'point_pools',read(root/f"configs/A/{r['kind']}.json"),g):(r,g) for r,g in jobs}
        for future in as_completed(future_map):
            record,g=future_map[future]
            try:
                future.result();cfg=read(root/f"configs/A/{record['kind']}.json")
                finished.append(pool_audit(root,record,cfg,g))
            except Exception as e:failed.append(dict(kind=record['kind'],generation=g,error=repr(e)))
            write(root/'PRECACHE_STATUS.json',dict(completed=len(finished),total=100,failed=failed))
            print(f"Precache {len(finished)}/100; failures {len(failed)}",flush=True)
    if failed:raise RuntimeError('Solid pool generation failed; retaining every failure')
    assert len(finished)==100
    write(root/'PRECACHE_COMPLETE.json',dict(pools=100,full_file_hashes_verified=True,records=finished))


def milestone(out,step,checkpoint,cases,plan,device):
    target=out/'fit'/f'step_{step:05d}';target.mkdir(parents=True,exist_ok=True)
    stable=target/'checkpoint.pt';shutil.copyfile(checkpoint,stable)
    evaluate(stable,cases[0]['path'],target/'diagnostics',device,plan['resolution'],plan['mesh_resolution'],observed_only=True)
    write(target/'VISUALIZATION_COMPLETE.json',dict(step=step,checkpoint_sha256=digest(stable),observed_only=True))


def run_arm(root,arm,device):
    root,plan=verify(root)
    assert read(root/'PRECACHE_COMPLETE.json')['pools']==100
    assert (root/'gates/GPU_GATE_COMPLETE.json').exists()
    failures=[]
    for task in [t for t in plan['tasks'] if t['arm']==arm]:
        out=root/task['output'];case=root/task['case'];cfg=read(root/task['config']);kind=task['kind']
        try:
            checkpoint=fit([str(case)],[None],cfg,out/'fit',device,steps=4000,pretrained=None,resume=False,
                milestones=plan['milestones'],diagnose=lambda step,ck,cases:milestone(out,step,ck,cases,plan,device))
            metric=evaluate(checkpoint,case,out/'final_evaluation',device,plan['resolution'],plan['mesh_resolution'],
                heldout_contours=root/f'evaluation_only/{kind}/heldout_contours.json',domain=root/f'evaluation_only/{kind}/evaluation_domain.json')
            write(out/'RUN_COMPLETE.json',dict(arm=arm,kind=kind,steps=4000,observed_planes=25,
                checkpoint_sha256=digest(checkpoint),observed_mean_dice=metric['observed_mean_dice'],
                strict_heldout_mean_dice=metric['strict_heldout_mean_dice'],heldout_used_for_training_or_selection=False))
        except Exception as e:
            import traceback
            failures.append(kind);write(out/'RUN_FAILED.json',dict(task=task,error=repr(e),traceback=traceback.format_exc()))
            print(f'FAILED {arm}/{kind}: {e}',flush=True)
    write(root/f'ARM_{arm}_COMPLETE.json',dict(arm=arm,attempted_kinds=KINDS,failed=failures))
    with (root/'summary.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        if all((root/f'ARM_{a}_COMPLETE.json').exists() for a in ARMS):
            results=[];missing=[]
            for task in plan['tasks']:
                done=root/task['output']/'RUN_COMPLETE.json'
                if done.exists():results.append(read(done))
                else:missing.append(dict(arm=task['arm'],kind=task['kind']))
            write(root/'SUMMARY.json',dict(results=results,missing_or_failed=missing,full_is_same_seed_repeat_of_D=True,
                primary='fixed final step4000 strict heldout section Dice; observed Dice also retained',
                heldout_used_for_training_or_selection=False))
            if not missing:write(root/'SUITE_COMPLETE.json',dict(models=10,all_cases_complete=True,steps=4000))
    if failures:raise RuntimeError(f'Failed {arm}: {failures}')


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--repository',required=True);p.add_argument('--suite',required=True)
    p=sub.add_parser('precache');p.add_argument('--suite',required=True);p.add_argument('--device',default='cuda');p.add_argument('--workers',type=int,default=2)
    p=sub.add_parser('arm');p.add_argument('--suite',required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--device',default='cuda')
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.repository,a.suite)
    elif a.command=='precache':precache(a.suite,a.device,a.workers)
    else:run_arm(a.suite,a.arm,a.device)


if __name__=='__main__':main()
