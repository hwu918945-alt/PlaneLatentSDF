"""Create a portable paired-patient subset without changing the fixed split."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data import load_case, validate_split


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/plane_sdf_shell'))
    parser.add_argument('--output', type=Path, default=Path('preflight/single_patient'))
    parser.add_argument('--case-id', default='002')
    parser.add_argument('--validation-case-id', default='012')
    args = parser.parse_args()
    source, destination = args.data.resolve(), args.output.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Subset output must be separate from the original cache')
    pairs = [('train', args.case_id), ('val', args.validation_case_id)]
    pending, split_paths = [], {'train': [], 'val': []}
    for split, patient in pairs:
        if not patient or Path(patient).name != patient or patient in ['.', '..']:
            raise ValueError('Invalid patient ID')
        for phase, index in [('ED', 0), ('ES', 1)]:
            original = source / split / f'{patient}_{phase}.npz'
            case = load_case(original)
            if case['case_id'] != patient or case['phase'] != index:
                raise ValueError(f'Patient/phase metadata mismatch: {original}')
            target = destination / split / original.name
            sha = digest(original)
            if target.exists() and digest(target) != sha:
                raise ValueError(f'Refusing to replace a different cache: {target}')
            pending.append((original, target, sha))
            split_paths[split].append(original)
    validate_split(split_paths)
    expected = {target for _, target, _ in pending}
    extra = set(destination.glob('*/*.npz')) - expected
    if extra:
        raise ValueError(f'Subset already contains other patients: {sorted(map(str, extra))}')
    records = []
    for original, target, sha in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(original, target)
        records.append({'source': f'{original.parent.name}/{original.name}',
                        'subset': str(target.relative_to(destination)), 'sha256': sha})
    report = {'status': 'PASS', 'train_patient': args.case_id,
              'validation_patient': args.validation_case_id, 'files': records,
              'test_used_for_fitting': False}
    (destination / 'subset_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
