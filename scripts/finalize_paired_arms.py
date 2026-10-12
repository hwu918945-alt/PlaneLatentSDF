"""Publish the five-arm summary after all five arm jobs have attempted their fits."""
import argparse
import fcntl
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ARMS = ('A', 'B', 'C', 'D', 'FULL')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--failed', type=int, nargs='*', default=[])
    args = parser.parse_args()
    root = Path(args.suite).resolve()
    sys.path.insert(0, str(root / 'runtime'))
    from plane_auto_occ.paired_suite import verify, summarize
    verify(root)
    plan = json.loads((root / 'plan.json').read_text())
    indices = [t['task_index'] for t in plan['tasks'] if t['arm'] == args.arm]
    assert len(indices) == 20 and set(args.failed) <= set(indices)
    atomic_json(root / f'ARM_{args.arm}_COMPLETE.json', dict(
        arm=args.arm, attempted_indices=indices, failed_indices=args.failed,
        created_utc=datetime.now(timezone.utc).isoformat()))
    with (root / 'SUMMARY.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not all((root / f'ARM_{arm}_COMPLETE.json').exists() for arm in ARMS):
            return
        arms = [json.loads((root / f'ARM_{arm}_COMPLETE.json').read_text()) for arm in ARMS]
        assert sorted(i for arm in arms for i in arm['attempted_indices']) == list(range(100))
        summary = summarize(root)
        completed = sum(row['completed'] for row in summary['arms'].values())
        complete = completed == 100 and not any(arm['failed_indices'] for arm in arms)
        atomic_json(root / ('SUITE_COMPLETE.json' if complete else 'SUITE_INCOMPLETE.json'), dict(
            arms=arms, completed_stage_models=completed, expected_stage_models=100,
            all_tasks_attempted=True, all_results_present=complete,
            failures_kept_in_summary=True, NO_ZERO_CROSSING_kept_as_scored_result=True,
            independent_generalization_claim=False, full_is_independent_repeat_of_D=True,
            created_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
