"""Verify the published files, historical method locks and cache contracts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_files(manifest):
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Published path resolves outside repository: {relative}')
        if expected['kind'] == 'symlink':
            if not path.is_symlink() or str(path.readlink()) != expected['target'] or not path.exists():
                raise ValueError(f'Broken or changed relative symlink: {relative}')
        elif not path.is_file() or path.is_symlink() or digest(path) != expected['sha256']:
            raise ValueError(f'Missing or changed published file: {relative}')
    # provenance/code_sha256.txt records the historical epi run, not the
    # intentionally revised main method. Current source hashes are in manifest.
    experiments = {}
    for name in ['stanford_bunny_slice_sweep_v2', 'balloon_dog_slice_sweep_v1']:
        folder = ROOT / 'experiments' / name
        locked = json.loads((folder / 'manifest.json').read_text())['sha256']
        for relative, sha in locked.items():
            if digest(folder / relative) != sha:
                raise ValueError(f'Changed experimental protocol/input: {name}/{relative}')
        experiments[name] = len(locked)
    folder = ROOT / 'experiments/stanford_bunny_single_case_v1'
    snapshot = json.loads((folder / 'runtime/source_snapshot.json').read_text())
    for record in snapshot:
        if digest(folder / record['snapshot']) != record['sha256']:
            raise ValueError(f'Changed Bunny v1 runtime: {record["snapshot"]}')
    experiments['stanford_bunny_single_case_v1'] = len(snapshot)
    return {'published_files': len(manifest['files']), 'experiment_locks': experiments,
            'current_method': 'SAX-only myocardium shell', 'historical_experiment_runtimes_unchanged': True}


def verify_caches():
    import numpy as np
    from core.data import load_case, validate_split

    folder = ROOT / 'data/plane_sdf'
    paths = {split: sorted((folder / split).glob('*.npz')) for split in ['train', 'val', 'test']}
    validate_split(paths)
    report = {}
    for split, expected in [('train', 285), ('val', 36), ('test', 36)]:
        phases = {}
        for path in paths[split]:
            case = load_case(path)
            label = 'ED' if case['phase'] == 0 else 'ES'
            if path.stem != f'{case["case_id"]}_{label}':
                raise ValueError(f'Filename/metadata mismatch: {path}')
            if case['phase'] in phases.setdefault(case['case_id'], set()):
                raise ValueError(f'Duplicate patient phase: {path}')
            phases[case['case_id']].add(case['phase'])
            distances = case['sdf2d'].numpy()
            if case['scale_mm'] != 100.0 or len(case['s']) < 3:
                raise ValueError(f'Clinical scale/plane count mismatch: {path}')
            if case['q'].shape[1] != 1024 or case['boundary'].shape[1] != 256:
                raise ValueError(f'Sampling contract mismatch: {path}')
            if not (np.any(distances < 0) and np.any(distances > 0)):
                raise ValueError(f'Missing inside/outside labels: {path}')
        if len(phases) != expected or len(paths[split]) != 2 * expected:
            raise ValueError(f'Fixed split count mismatch: {split}')
        if any(value != {0, 1} for value in phases.values()):
            raise ValueError(f'Missing paired ED/ES: {split}')
        report[split] = {'patients': len(phases), 'phase_files': len(paths[split])}
    synthetic = []
    for path in sorted((ROOT / 'experiments').glob('*/data/*observed.npz')) + sorted(
            (ROOT / 'experiments').glob('*/slices_*/data/*observed.npz')):
        case = load_case(path)
        if case['scale_mm'] != 100.0 or case['phase'] != 0:
            raise ValueError(f'Benchmark normalization mismatch: {path}')
        if path.parent.parent.name.startswith('slices_'):
            expected = int(path.parent.parent.name.split('_')[1])
            if len(case['s']) != expected:
                raise ValueError(f'Observed plane count mismatch: {path}')
        synthetic.append({'file': str(path.relative_to(ROOT)), 'planes': len(case['s'])})
    if len(synthetic) != 9:
        raise ValueError(f'Expected Bunny v1 plus eight sweep caches; found {len(synthetic)}')
    return {'clinical': report, 'synthetic': synthetic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hashes-only', action='store_true', help='Skip NumPy/PyTorch cache loading')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'provenance/release/manifest.json').read_text())
    report = verify_files(manifest)
    if not args.hashes_only:
        report['caches'] = verify_caches()
    report['status'] = 'PASS'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
