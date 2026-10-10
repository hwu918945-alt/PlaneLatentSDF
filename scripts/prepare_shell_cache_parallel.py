"""Parallel deployment wrapper around the imported, unchanged shell converter."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_one(task):
    import numpy as np
    from prepare_cache import convert
    from core.data import load_case, require_shell
    from core.sampling import SCHEMA
    record, output, seed, projection, resume = task
    key = (record['case_id'], record['phase'])
    name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in record['case_id'])
    target = Path(output) / record['split'] / f'{name}_{record["phase"]}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    signature = dict(record_sha256=hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                     seed=seed, project_parallel=projection, schema=SCHEMA)
    sidecar = target.with_suffix('.input.json')
    if target.exists():
        if not resume or not sidecar.exists() or json.loads(sidecar.read_text()) != signature:
            raise ValueError(f'Refusing unmatched existing cache: {target}')
        require_shell(load_case(target))
        return dict(file=str(target), resumed=True)
    entropy = int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16], 16)
    # Converter and RNG are identical to prepare_cache.py. Only scheduling and
    # atomic placement differ; concurrent tasks never share patient/phase files.
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=target.parent) as folder:
        temporary = Path(folder) / target.name
        report = convert(record, temporary, np.random.default_rng(entropy), projection)
        require_shell(load_case(temporary))
        # A signature without a final NPZ can safely be regenerated after an
        # interruption. Place it first so a committed NPZ always has its input lock.
        sidecar.write_text(json.dumps(signature, indent=2) + '\n')
        temporary.replace(target)
    report['file'] = str(target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/aligned_sax_shell_manifest.json'))
    parser.add_argument('--output', type=Path, default=Path('data/plane_sdf_shell'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--project-parallel', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = '1'
    records = json.loads(args.manifest.read_text())['records']
    seen, splits = set(), {}
    for record in records:
        key = (record['case_id'], record['phase'])
        if key in seen or record['phase'] not in ['ED', 'ES'] or record['split'] not in ['train', 'val', 'test']:
            raise ValueError(f'Duplicate or invalid identity: {key}')
        seen.add(key)
        if splits.setdefault(record['case_id'], record['split']) != record['split']:
            raise ValueError('Patient split leakage')
    tasks = [(record, str(args.output), args.seed, args.project_parallel, args.resume) for record in records]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(prepare_one, task) for task in tasks]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(f'{len(reports)}/{len(records)} {report["file"]}', flush=True)
    reports.sort(key=lambda row: row['file'])
    (args.output / 'preparation_report.json').write_text(json.dumps({
        'schema': 'sax_myo_shell_v1', 'workers': args.workers,
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'records': reports}, indent=2) + '\n')


if __name__ == '__main__':
    main()
