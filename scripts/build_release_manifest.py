"""Refresh release hashes from explicitly staged code and prepared input files."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'provenance/release/manifest.json'


def main():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')[:-1]
    entries = {}
    for relative in files:
        if relative == OUTPUT:
            continue
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'External published path: {relative}')
        if any(part in ['runs', 'outputs', 'logs', 'reports', 'results', 'preflight'] for part in Path(relative).parts):
            raise ValueError(f'Runtime/result artifact staged: {relative}')
        if path.suffix in ['.pt', '.pth', '.ckpt', '.out', '.err', '.log']:
            raise ValueError(f'Unwanted runtime artifact staged: {relative}')
        if path.is_symlink():
            if path.readlink().is_absolute() or not path.exists():
                raise ValueError(f'Nonportable symlink: {relative}')
            entries[relative] = {'kind': 'symlink', 'target': str(path.readlink())}
        else:
            size = path.stat().st_size
            if size >= 100 * 1024 * 1024:
                raise ValueError(f'File exceeds GitHub individual-file limit: {relative}')
            entries[relative] = {'kind': 'file', 'bytes': size,
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    cache_count = sum(p.startswith('data/plane_sdf_shell/') and p.endswith('.npz') for p in entries)
    if cache_count != 714 or 'data/plane_sdf_shell/DATASET_READY.json' not in entries:
        raise ValueError('Stage all 714 audited shell caches before publishing')
    manifest = {'format_version': 1, 'publication_date': '2026-10-10',
                'scope': 'SAX-only myocardium shell code and prepared inputs; no numerical run reports',
                'method': 'SAX-only myocardium shell',
                'imported_method_commit': '24c0be0df78b98879fe1b6b35a6f22745e88ba45',
                'excluded_from_self_hash': [OUTPUT],
                'file_count': len(entries),
                'total_regular_file_bytes': sum(row.get('bytes', 0) for row in entries.values()),
                'clinical_shell_cache': {'train': {'patients': 285, 'phase_files': 570},
                                         'val': {'patients': 36, 'phase_files': 72},
                                         'test': {'patients': 36, 'phase_files': 72}},
                'files': dict(sorted(entries.items()))}
    (ROOT / OUTPUT).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': len(entries), 'shell_npz': cache_count,
                      'megabytes': round(manifest['total_regular_file_bytes'] / 1024**2, 2)}, indent=2))


if __name__ == '__main__':
    main()
