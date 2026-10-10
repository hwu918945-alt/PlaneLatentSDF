"""Check that the audited shell dataset still matches its frozen input inventory."""
import argparse
import hashlib
import json
from pathlib import Path


def check(data, manifest):
    ready = json.loads((data / 'DATASET_READY.json').read_text())
    if ready['status'] != 'PASS' or ready['schema'] != 'sax_myo_shell_v1' or not ready['continuous_source_polygons_audited']:
        raise ValueError('Missing completed continuous-source shell audit')
    if ready['patients'] != {'train': 285, 'val': 36, 'test': 36} or ready['phase_files'] != {'train': 570, 'val': 72, 'test': 72}:
        raise ValueError('Unexpected fixed patient/phase split')
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != ready['manifest_sha256']:
        raise ValueError('Aligned shell inputs changed after the dataset audit')
    actual = {str(path.relative_to(data)) for path in data.glob('*/*.npz')}
    if actual != set(ready['files']) or len(actual) != 714:
        raise ValueError('Prepared shell cache inventory changed')
    for relative, sha in ready['files'].items():
        if hashlib.sha256((data / relative).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Changed audited shell cache: {relative}')
    return {'status': 'PASS', 'patients': ready['patients'], 'phase_files': ready['phase_files'],
            'schema': ready['schema'], 'continuous_source_polygons_audited': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/plane_sdf_shell'))
    parser.add_argument('--manifest', type=Path, default=Path('data/aligned_sax_shell_manifest.json'))
    args = parser.parse_args()
    print(json.dumps(check(args.data, args.manifest), indent=2))


if __name__ == '__main__':
    main()
