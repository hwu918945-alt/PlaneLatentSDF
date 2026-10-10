"""Shard the original continuous-polygon audit by patient, then enforce full coverage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/plane_sdf_shell'))
    parser.add_argument('--manifest', type=Path, default=Path('data/aligned_sax_shell_manifest.json'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('preflight/shell_dataset_audit.json'))
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    data = args.data.resolve()
    document = json.loads(args.manifest.read_text())
    patients = {}
    identities = set()
    for record in document['records']:
        key = (record['case_id'], record['phase'])
        if key in identities:
            raise ValueError('Duplicate source patient/phase')
        identities.add(key)
        patients.setdefault(record['case_id'], []).append(record)
    for patient, records in patients.items():
        if len(records) != 2 or {r['phase'] for r in records} != {'ED', 'ES'} or len({r['split'] for r in records}) != 1:
            raise ValueError(f'Invalid paired patient/split: {patient}')
    buckets = [[] for _ in range(min(args.workers, len(patients)))]
    # Balance by actual plane count while retaining both phases of each patient.
    sizes = [0] * len(buckets)
    for records in sorted(patients.values(), key=lambda rows: sum(len(r['planes']) for r in rows), reverse=True):
        index = sizes.index(min(sizes))
        buckets[index].extend(records)
        sizes[index] += sum(len(r['planes']) for r in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='.shell-audit-', dir=args.output.parent) as folder:
        commands = []
        outputs = []
        for index, records in enumerate(buckets):
            shard = Path(folder) / str(index)
            shard.mkdir()
            manifest = shard / 'manifest.json'
            manifest.write_text(json.dumps(dict(document, records=records)))
            for record in records:
                original = data / record['split'] / f'{record["case_id"]}_{record["phase"]}.npz'
                target = shard / 'data' / record['split'] / original.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(original)
            output = shard / 'audit.json'
            outputs.append(output)
            commands.append([sys.executable, str(ROOT / 'scripts/verify_dataset.py'),
                             '--data', str(shard / 'data'), '--manifest', str(manifest),
                             '--allow-subset', '--output', str(output)])
        def run(command):
            result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stderr + result.stdout)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run, commands))
        reports = [json.loads(path.read_text()) for path in outputs]
    audited = [row for report in reports for row in report['records']]
    actual = {(row['case_id'], 'ED' if row['phase'] == 0 else 'ES') for row in audited}
    if actual != identities or len(audited) != len(identities):
        raise ValueError('Parallel audit did not cover every source phase exactly once')
    paths = {(record['case_id'], record['phase']):
             data / record['split'] / f'{record["case_id"]}_{record["phase"]}.npz'
             for record in document['records']}
    expected_paths = set(paths.values())
    if set(data.glob('*/*.npz')) != expected_paths:
        raise ValueError('Prepared cache inventory differs from the source manifest')
    for row in audited:
        row['file'] = str(paths[(row['case_id'], 'ED' if row['phase'] == 0 else 'ES')])
    counts = {split: len({r['case_id'] for r in document['records'] if r['split'] == split})
              for split in ['train', 'val', 'test']}
    if counts != {'train': 285, 'val': 36, 'test': 36}:
        raise ValueError(f'Fixed patient split mismatch: {counts}')
    report = dict(status='PASS', patients=counts,
                  phase_files={split: 2 * count for split, count in counts.items()},
                  max_plane_residual_mm=max(r['max_plane_residual_mm'] for r in reports),
                  max_sdf_error_mm=max(r['max_sdf_error_mm'] for r in reports),
                  source_counts=reports[0]['source_counts'], subset_only=False,
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  workers=args.workers, records=sorted(audited, key=lambda row: row['file']))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    ready = {'status': 'PASS', 'schema': 'sax_myo_shell_v1', 'anatomy': 'LV_myocardium_shell',
             'scale_mm': 100.0, 'patients': counts, 'phase_files': report['phase_files'],
             'manifest_sha256': report['manifest_sha256'],
             'continuous_source_polygons_audited': True,
             'files': {str(path.relative_to(data)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(expected_paths)}}
    (data / 'DATASET_READY.json').write_text(json.dumps(ready, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2))


if __name__ == '__main__':
    main()
