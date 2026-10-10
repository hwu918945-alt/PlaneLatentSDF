import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--slices',type=int,required=True);args=p.parse_args()
    group=ROOT/f'slices_{args.slices:03d}'
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for relative,sha in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==sha,f'Changed locked input: {relative}'
    assert json.loads((group/'data/verification.json').read_text())['status']=='PASS'
    subprocess.run([sys.executable,str(ROOT/'scripts/fit_sweep.py'),'--slices',str(args.slices),'--resume'],check=True,cwd=group)
    for step in [2000,4000,6000]:
        out=group/f'outputs/step_{step:06d}';out.mkdir(parents=True,exist_ok=True)
        if (out/'EVALUATION_COMPLETE.json').exists():continue
        with (out/'eval.out').open('w') as stdout,(out/'eval.err').open('w') as stderr:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_bunny.py'),'--checkpoint',f'runs/main/step_{step:06d}.pt',
                '--output',str(out),'--domain',str(ROOT/'data/evaluation_domain.json')],cwd=group,stdout=stdout,stderr=stderr)
        if result.returncode:
            (out/'EVALUATION_FAILED.json').write_text(json.dumps(dict(step=step,returncode=result.returncode,
                successful_reconstruction_claimed=False,details='eval.err'),indent=2))
        else:
            metrics=json.loads((out/'EVALUATION_COMPLETE.json').read_text())
            assert metrics['observed_planes']==args.slices and metrics['optimization_steps']==step
            assert metrics['heldout_planes']==121
    failures=[step for step in [2000,4000,6000] if not (group/f'outputs/step_{step:06d}/EVALUATION_COMPLETE.json').exists()]
    (group/'GROUP_COMPLETE.json').write_text(json.dumps(dict(observed_planes=args.slices,steps=[2000,4000,6000],
        evaluation_failures=failures,slurm_job_id=os.environ.get('SLURM_JOB_ID'),execution_complete=not failures),indent=2))
    if failures:raise RuntimeError(f'Preserved evaluation failures at {failures}')


if __name__=='__main__':main()
