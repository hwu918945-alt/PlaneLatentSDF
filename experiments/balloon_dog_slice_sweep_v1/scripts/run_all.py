"""Run all prespecified groups and aggregate within a single Slurm job."""
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());statuses=[]
    for count in protocol['slice_counts']:
        print(f'Start Balloon Dog group: {count} observed slices',flush=True)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_group.py'),'--slices',str(count)],cwd=ROOT)
        statuses.append(dict(observed_planes=count,returncode=result.returncode))
        if result.returncode:
            failed=ROOT/f'slices_{count:03d}/PIPELINE_FAILED.json'
            failed.write_text(json.dumps(dict(observed_planes=count,returncode=result.returncode,
                execution_complete=False,successful_reconstruction_claimed=False,details='Slurm logs and per-checkpoint eval.err'),indent=2))
    summary=subprocess.run([sys.executable,str(ROOT/'scripts/aggregate.py')],cwd=ROOT)
    report=dict(slurm_job_id=os.environ.get('SLURM_JOB_ID'),groups=statuses,summary_returncode=summary.returncode,
        execution_complete=all(s['returncode']==0 for s in statuses) and summary.returncode==0,
        scientific_reconstruction_success_claimed=False)
    (ROOT/'PIPELINE_RUN_COMPLETE.json').write_text(json.dumps(report,indent=2))
    if not report['execution_complete']:raise SystemExit(1)


if __name__=='__main__':main()
