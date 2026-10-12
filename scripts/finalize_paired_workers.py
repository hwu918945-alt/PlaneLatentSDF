"""Publish one five-arm summary after all four workers have attempted their tasks."""
import argparse
import fcntl
import json
import sys
from datetime import datetime,timezone
from pathlib import Path


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--worker',type=int,required=True)
    p.add_argument('--failed',type=int,nargs='*',default=[]);a=p.parse_args()
    root=Path(a.suite).resolve();assert 0<=a.worker<4
    indices=list(range(a.worker*25,(a.worker+1)*25));assert set(a.failed)<=set(indices)
    atomic_json(root/f'WORKER_{a.worker}_COMPLETE.json',dict(worker=a.worker,attempted_indices=indices,
        failed_indices=a.failed,created_utc=datetime.now(timezone.utc).isoformat()))
    with (root/'SUMMARY.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not all((root/f'WORKER_{i}_COMPLETE.json').exists() for i in range(4)):return
        workers=[json.loads((root/f'WORKER_{i}_COMPLETE.json').read_text()) for i in range(4)]
        assert sorted(j for w in workers for j in w['attempted_indices'])==list(range(100))
        sys.path.insert(0,str(root/'runtime'))
        from plane_auto_occ.paired_suite import verify,summarize
        verify(root);summary=summarize(root)
        completed=sum(r['completed'] for r in summary['arms'].values())
        complete=completed==100 and not any(w['failed_indices'] for w in workers)
        atomic_json(root/('SUITE_COMPLETE.json' if complete else 'SUITE_INCOMPLETE.json'),dict(
            workers=workers,completed_stage_models=completed,expected_stage_models=100,
            all_tasks_attempted=True,all_results_present=complete,
            failures_kept_in_summary=True,NO_ZERO_CROSSING_kept_as_scored_result=True,
            independent_generalization_claim=False,full_is_independent_repeat_of_D=True,
            created_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
