#!/usr/bin/env python3
"""Submit offline CPU preprocessing, then GPU training with afterok dependency."""
import argparse,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--account',default='xiaowusun_comp');a=p.parse_args()
    root=Path(a.experiment).resolve();runtime=root/'runtime';cfg=json.loads((root/'config.json').read_text())
    if cfg.get('pipeline_backend')!='offline_batched_v1':raise ValueError('Requires offline batched config')
    if not (root/'runtime_lock.json').exists():raise ValueError('Freeze committed source before submission')
    (root/'logs').mkdir(exist_ok=True);common=['sbatch','--parsable',f'--account={a.account}',f'--export=ALL,TASK_EXPERIMENT={root}']
    cpu=subprocess.check_output(common+['--job-name=auto_occ_precache',f'--output={root}/logs/precache_%j.out',f'--error={root}/logs/precache_%j.err',str(runtime/'scripts/precache_job.sh')],text=True).strip().split(';')[0]
    gpu=subprocess.check_output(common+[f'--dependency=afterok:{cpu}','--job-name=auto_occ_heart_fast',f'--output={root}/logs/auto_occ_heart_fast_%j.out',f'--error={root}/logs/auto_occ_heart_fast_%j.err',str(runtime/'scripts/experiment_job.sh')],text=True).strip().split(';')[0]
    report=dict(precache_job=cpu,training_job=gpu,dependency='afterok',cache_git_tracked=False)
    (root/'SUBMISSION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
