"""Build/install in isolation, test portable runs, and reproduce fixed V2 metrics."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('TASK_MIGRATION_VALIDATION_OUTPUT',str(ROOT/'runs/migration_validation_20261011'))).resolve()
OUT.mkdir(parents=True,exist_ok=True)
LOGS=OUT/'logs';LOGS.mkdir(exist_ok=True)
provenance=json.loads((ROOT/'docs/MIGRATION_PROVENANCE.json').read_text())
protected=['model.py','objective.py','geometry.py','data.py','inference.py','evaluate.py',
           'extract_mesh.py','area_pool.py','chunked.py','config.json','make_synthetic.py','tests.py']
for name in protected:
    assert hashlib.sha256((ROOT/'plane_auto_occ'/name).read_bytes()).hexdigest()==provenance['original_source_sha256'][name],name

def run(name,args,cwd=ROOT,env=None):
    print(name,flush=True)
    with (LOGS/(name+'.out')).open('w') as out,(LOGS/(name+'.err')).open('w') as err:
        r=subprocess.run([sys.executable,'-B',*map(str,args)],cwd=cwd,env=env,stdout=out,stderr=err)
    if r.returncode:
        raise RuntimeError(f'{name} failed; see {LOGS/(name+".err")}')
    counts={'baseline_tests':6,'repair_tests':6,'migration_tests':2}
    if name in counts:
        match=re.search(r'Ran (\d+) tests?',(LOGS/(name+'.err')).read_text())
        assert match and int(match.group(1))==counts[name],f'{name}: test discovery/count mismatch'

wheels=OUT/'wheels';wheels.mkdir(exist_ok=True)
run('build_wheel',['-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',wheels,ROOT])
wheel=next(wheels.glob('plane_auto_occupancy-*.whl'))
installed=OUT/'installed'
if installed.exists():raise FileExistsError('Validation install directory must be new')
run('install_wheel',['-m','pip','install','--no-deps','--target',installed,wheel])
env=dict(os.environ,PYTHONPATH=str(installed),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
run('installed_help',['-m','plane_auto_occ','--help'],cwd=OUT,env=env)
run('baseline_tests',['-m','plane_auto_occ','test'],cwd=OUT,env=env)
run('repair_tests',['-m','plane_auto_occ','test-repair'],cwd=OUT,env=env)
run('migration_tests',['-m','plane_auto_occ','test-migration'],cwd=OUT,env=env)

worker=OUT/'isolated_worker.py'
worker.write_text('''import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
out=Path(sys.argv[1]).resolve();root=Path(sys.argv[2]).resolve()
forbidden=str(root.parent/'PlaneLatentSDF')
def guard(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes)):
        name=args[0].decode() if isinstance(args[0],bytes) else args[0]
        if name.startswith(forbidden):raise RuntimeError('Forbidden original-repository read: '+name)
sys.addaudithook(guard)
import plane_auto_occ
assert Path(plane_auto_occ.__file__).resolve().is_relative_to(out/'installed')
from plane_auto_occ.inference import restore,bounds,plane_grid,query
from plane_auto_occ.geometry import contour_labels
from plane_auto_occ.paths import resolve_sampling_pool
from plane_auto_occ.deployment import verify_sources
torch.set_num_threads(2)
exp=root/'runs/plane_auto_occ_repair_002_ED_20261011_v2'
verify_sources(exp)
checkpoint=exp/'outputs/fit/last.pt';case=exp/'cache/train/002_ED.npz'
model,code,u,c,full=restore(checkpoint,case)
lo,hi,basis=bounds(c,5)
native=json.loads((exp/'outputs/final_eval/metrics.json').read_text())
rows=[]
for i,pos in enumerate(full['s'].numpy()):
    q=plane_grid(float(pos),lo,hi,basis,160);logits=query(model,q,c,u)
    reg,_,_=contour_labels(q,full['boundary'][i].numpy(),full['boundary_endo'][i].numpy(),basis)
    pred=logits>=0;gt=reg==0
    dice=float((2*(pred&gt).sum()+1e-6)/(pred.sum()+gt.sum()+1e-6))
    expected=native['planes'][i]['contour_proxy_dice']
    assert abs(dice-expected)<1e-6,(i,dice,expected)
    rows.append(dict(index=i,dice=dice,original_dice=expected))
pools=[]
for f in sorted((exp/'outputs/fit').glob('step_*/checkpoint.pt'))+[checkpoint]:
    ck=torch.load(f,map_location='cpu',weights_only=False)
    pool=resolve_sampling_pool(f,ck['sampling_pool'])
    assert pool.is_relative_to(exp)
    pools.append(str(pool.relative_to(exp)))
report=dict(installed_package=str(Path(plane_auto_occ.__file__).resolve()),
    original_repository_reads_blocked=True,all_7_final_dice_reproduced=True,
    source_lock_verified=True,milestone_pools_resolved_locally=True,planes=rows,pools=pools)
(out/'relocated_checkpoint_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
'''.replace('pred&gt','pred & gt'))
run('relocated_checkpoint',[worker,OUT,ROOT],cwd=OUT,env=env)
run('synthetic_raw',['-m','plane_auto_occ','make-synthetic','--output',OUT/'synthetic_raw'],cwd=OUT,env=env)
run('synthetic_conversion',['-m','plane_auto_occ','convert-cache','--input-root',OUT/'synthetic_raw','--output-root',OUT/'synthetic_cache'],cwd=OUT,env=env)
cfg=json.loads((ROOT/'plane_auto_occ/config.json').read_text())
cfg.update(points=128,microbatch_points=32,width=32,steps=2,pool_capacities=[512,256,256])
config=OUT/'synthetic_config.json';config.write_text(json.dumps(cfg,indent=2))
prepared=OUT/'synthetic_prepared'
run('prepare_synthetic',['-m','plane_auto_occ','prepare-run','--case',OUT/'synthetic_cache/train/train_ED.npz',
    '--config',config,'--output',prepared],cwd=OUT,env=env)
run('deploy_synthetic',['-m','plane_auto_occ','deploy','--experiment',prepared,'--device','cpu'],cwd=OUT,env=env)
complete=json.loads((prepared/'RUN_COMPLETE.json').read_text());assert complete['steps']==2
result=dict(standalone_validation_complete=True,protected_source_files_byte_identical=protected,
    baseline_tests=6,repair_tests=6,migration_tests=2,total_tests=14,
    wheel_built_and_installed=True,installed_config_available=True,
    relocated_final_checkpoint=json.loads((OUT/'relocated_checkpoint_validation.json').read_text()),
    synthetic_pipeline=dict(device='cpu',cold_smoke_steps=2,formal_steps=2,run_complete=True,
        purpose='execution/portability validation; not clinical training'),
    slurm_jobs_submitted=0,old_repository_modified=False)
(ROOT/'docs/VALIDATION_STANDALONE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='relocated_final_checkpoint'},indent=2),flush=True)
