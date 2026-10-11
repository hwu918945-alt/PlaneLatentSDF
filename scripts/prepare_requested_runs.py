#!/usr/bin/env python3
"""Copy reusable contour caches into independent, ignored experiment directories."""
import argparse,json,hashlib,shutil,sys
from pathlib import Path
import numpy as np
from plane_auto_occ.solid import convert as convert_solid
from plane_auto_occ.convert_population_cache import convert
from plane_auto_occ.area_pool import digest


def locked_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists(): shutil.copy2(src,dst)
    if digest(src)!=digest(dst): raise ValueError(f'Copy mismatch: {src}')
    return dict(source=str(src),local=str(dst),sha256=digest(dst))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--repo-root',required=True)
    p.add_argument('--tag',default='20261011_v1');a=p.parse_args()
    src=Path(a.source_root).resolve();repo=Path(a.repo_root).resolve();base=json.loads((repo/'plane_auto_occ/config.json').read_text())
    root=repo/'runs'/f'heart_full_{a.tag}';root.mkdir(parents=True,exist_ok=True);copied=[]
    for split in ('train','val','test'):
        for file in sorted((src/'data/plane_sdf_shell'/split).glob('*.npz')):
            local=root/'data/source_shell'/split/file.name;copied.append(locked_copy(file,local))
            dst=root/'data/occupancy'/split/file.name
            if not dst.exists():convert(local,dst)
    auditsrc=src/'data/aligned_sax_shell_manifest_audit.json'
    if auditsrc.exists():copied.append(locked_copy(auditsrc,root/'data/source_shell/source_alignment_audit.json'))
    counts={}; patientsets={}
    for split in ('train','val','test'):
        records=[]
        for file in sorted((root/'data/occupancy'/split).glob('*.npz')):
            with np.load(file) as z: records.append(dict(path=str(file.relative_to(root)),case_id=str(z['case_id']),phase=int(z['phase']),planes=len(z['s']),sha256=digest(file)))
        patientsets[split]={r['case_id'] for r in records};counts[split]=dict(patients=len(patientsets[split]),phase_records=len(records),records=records)
        if len({(r['case_id'],r['phase']) for r in records})!=len(records):raise ValueError('Duplicate records')
        if {(r['case_id'],r['phase']) for r in records}!={(c,p) for c in patientsets[split] for p in (0,1)}:raise ValueError('Missing phase')
    if [len(patientsets[s]) for s in ('train','val','test')]!=[285,36,36]:raise ValueError('Split count mismatch')
    if any(patientsets[x]&patientsets[y] for x,y in (('train','val'),('train','test'),('val','test'))):raise ValueError('Patient split leakage')
    (root/'data_lock.json').write_text(json.dumps(dict(source_copies=copied,splits=counts,cache_semantics='true Epi-minus-Endo myocardium shell',excluded_stress_patients=3),indent=2))
    cfg=dict(base);cfg.update(steps=114000,save_every=250,log_every=50,pipeline_backend='offline_batched_v1',
        fast_query_points=524288,offline_pool_root=str(root/'point_pools'),cache_workers=3,prefetch_cases=2,checkpoint_seconds=300,sampler_cache_cases=570)
    (root/'config.json').write_text(json.dumps(cfg,indent=2))
    plan=dict(kind='heart',train_epochs=200,updates_per_epoch=570,total_updates=114000,
        val_epochs=[100,200],inference_steps=2000,train_observed='all SAX',
        heldout_observed='first, middle, last SAX only',checkpoint_selection='val patient-mean strict-heldout-slice Dice',
        test_protocol='fresh codes, frozen best shared G/D; test once after validation selection',
        visual_epochs=[0,50,100,150,200],resolution=160,mesh_resolution=96,seed=19)
    (root/'plan.json').write_text(json.dumps(plan,indent=2))
    descriptions=[('bunny_25','stanford_bunny_slice_sweep_v2','bunny_observed.npz','stanford_bunny_single_case_v1/source/bunny_reference_mm.ply'),
        ('balloon_dog_25','balloon_dog_slice_sweep_v1','balloon_dog_observed.npz','balloon_dog_slice_sweep_v1/source/balloon_dog_reference_mm.ply')]
    for name,previous,filename,reference in descriptions:
        root=repo/'runs'/f'{name}_{a.tag}';root.mkdir(parents=True,exist_ok=True);copied=[]
        for srcfile,dst in [(src/'experiments'/previous/'slices_025/data'/filename,root/'data/source_observed.npz'),
            (src/'experiments'/previous/'slices_025/data/observed_contours.json',root/'data/observed_contours.json'),
            (src/'experiments'/previous/'data/heldout_contours.json',root/'evaluation_only/heldout_contours.json'),
            (src/'experiments'/previous/'data/evaluation_domain.json',root/'evaluation_only/evaluation_domain.json'),
            (src/'experiments'/reference,root/'evaluation_only/reference.ply')]:copied.append(locked_copy(srcfile,dst))
        dst=root/'data/case.npz'
        if not dst.exists():convert_solid(root/'data/source_observed.npz',root/'data/observed_contours.json',dst)
        cfg=dict(base);cfg.update(steps=4000)
        (root/'config.json').write_text(json.dumps(cfg,indent=2))
        (root/'plan.json').write_text(json.dumps(dict(kind='solid',steps=4000,observed_planes=25,initialization='random seed 19',
            milestones=[0,1000,2000,3000,4000],resolution=160,mesh_resolution=128,
            heldout_used_for_training_or_checkpoint_selection=False),indent=2))
        (root/'data_lock.json').write_text(json.dumps(dict(source_copies=copied,case_sha256=digest(dst),
            source_reference_used_for_loss=False,observed_planes=25),indent=2))
        print(root,flush=True)
    print('Preparation complete',flush=True)

if __name__=='__main__':main()
