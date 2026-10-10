"""Build audited SAX-only shell pools from continuous epi/endo polygons."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from core.data import polygon_sdf, resample_contour
from core.geometry import validate_shell, shell_sdf
from core.mesh import stack_basis
from core.sampling import POOL_COUNTS, SOURCE_NAMES, SCHEMA


def random_boundary(polygon, count, rng):
    a = np.asarray(polygon, float)
    delta = np.roll(a, -1, axis=0)-a
    lengths = np.linalg.norm(delta, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    t = rng.uniform(0, cumulative[-1], count)
    index = np.minimum(np.searchsorted(cumulative, t, side='right')-1, len(a)-1)
    return a[index] + ((t-cumulative[index])/lengths[index])[:, None]*delta[index]


def region_points(epi, endo, tag, count, rng):
    low, high = epi.min(0)-10., epi.max(0)+10.
    if tag == 5:
        low, high = endo.min(0), endo.max(0)
    parts, remaining = [], count
    for _ in range(1000):
        points = rng.uniform(low, high, (max(1024, min(8192, remaining*4)), 2))
        de, di = polygon_sdf(points, epi), polygon_sdf(points, endo)
        masks = {4: (de < -1e-6)&(di > 1e-6), 5: di < -1e-6, 6: de > 1e-6}
        valid = points[masks[tag]][:remaining]
        parts.append(valid); remaining -= len(valid)
        if remaining == 0:
            return np.concatenate(parts)
    raise ValueError(f'Insufficient region area for {SOURCE_NAMES[tag]}; no quota substitution allowed')


def plane_pool(epi, endo, rng):
    epi, endo = validate_shell(epi, endo)
    parts = []
    for tag, count in enumerate(POOL_COUNTS):
        if tag in (0, 2):
            points = resample_contour(epi if tag == 0 else endo, count).astype(float)
        elif tag in (1, 3):
            points = random_boundary(epi if tag == 1 else endo, count, rng)
            sigma = rng.choice([1., 3., 6.], count, p=[.4, .4, .2])
            points += rng.normal(size=(count, 2))*sigma[:, None]
        else:
            points = region_points(epi, endo, tag, count, rng)
        parts.append(points)
    points = np.concatenate(parts)
    source = np.repeat(np.arange(7, dtype=np.int8), POOL_COUNTS)
    sdf = shell_sdf(points, epi, endo)
    sdf[(source == 0)|(source == 2)] = 0.
    return points, sdf, source


def convert(record, output, rng, project_parallel=False):
    if record.get('anatomy') != 'LV_myocardium_shell':
        raise ValueError('Manifest lacks validated endo; legacy epi-only manifest cannot generate shell')
    planes = record['planes']
    if len(planes) < 3:
        raise ValueError('Need >=3 valid observed shell SAX planes for LOO')
    normals = []
    for plane in planes:
        u, v = np.asarray(plane['u'], float), np.asarray(plane['v'], float)
        if max(abs(np.linalg.norm(u)-1), abs(np.linalg.norm(v)-1), abs(u@v)) > 1e-4:
            raise ValueError('Plane bases must be orthonormal')
        normals.append(np.cross(u, v))
    axis = normals[0]
    oriented = [n if n@axis > 0 else -n for n in normals]
    axis = np.mean(oriented, axis=0); axis /= np.linalg.norm(axis)
    planes = sorted(planes, key=lambda p: np.asarray(p['origin_mm'])@axis)
    # An always-visible endpoint origin uses plane geometry only. No shape centroid
    # from the hidden plane, and no per-case extent-dependent coordinate scale.
    center = np.asarray(planes[0]['origin_mm'], float)
    scale = 100.
    basis = stack_basis(axis)
    prepared, max_projection = [], 0.
    for plane in planes:
        origin = np.asarray(plane['origin_mm'], float)
        u, v = np.asarray(plane['u'], float), np.asarray(plane['v'], float)
        polygons = []
        for key in ('epi_uv_mm', 'endo_uv_mm'):
            uv = np.asarray(plane[key], float)
            world = origin + uv[:, :1]*u + uv[:, 1:]*v
            residual = (world-origin)@axis
            projection = float(np.abs(residual).max())
            max_projection = max(max_projection, projection)
            if projection > .01 and not project_parallel:
                raise ValueError(f'Nonparallel stack requires --project-parallel: {projection:.4f} mm')
            polygons.append((world-origin)@basis[:, :2])
        epi, endo = validate_shell(*polygons)
        uv, sdf, source = plane_pool(epi, endo, rng)
        s_mm = (origin-center)@axis
        point_mm = origin + uv@basis[:, :2].T
        boundaries = [origin + resample_contour(poly, 512)@basis[:, :2].T for poly in (epi, endo)]
        prepared.append((s_mm/scale, (point_mm-center)/scale, sdf/scale, source,
                         (boundaries[0]-center)/scale, (boundaries[1]-center)/scale,
                         plane.get('source_slice_index', len(prepared))))
    s, q, sdf, source, epi, endo, ids = zip(*prepared)
    if np.any(np.diff(s) <= 1e-5):
        raise ValueError('Duplicate or unordered physical planes')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    np.savez_compressed(output, q=np.asarray(q, np.float32), sdf2d=np.asarray(sdf, np.float32),
                        sampling_source=np.asarray(source, np.int8), boundary=np.asarray(epi, np.float32),
                        boundary_endo=np.asarray(endo, np.float32), s=np.asarray(s, np.float32),
                        axis=np.asarray(axis, np.float32), center_mm=center, scale_mm=np.array(scale),
                        phase=np.array(0 if record['phase']=='ED' else 1), case_id=np.array(record['case_id']),
                        schema=np.array(SCHEMA), anatomy=np.array('LV_myocardium_shell'),
                        source_slice_indices=np.asarray(ids, np.int32),
                        provenance_json=np.array(json.dumps({k:v for k,v in record.items() if k!='planes'})))
    return dict(file=str(output), max_plane_projection_mm=max_projection,
                explicit_parallel_projection=project_parallel, center_source='first_valid_plane_origin',
                pool_counts=dict(zip(SOURCE_NAMES, POOL_COUNTS)), planes=len(s))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--project-parallel', action='store_true')
    p.add_argument('--seed', type=int, default=20261010)
    p.add_argument('--resume', action='store_true', help='Validate and reuse matching completed caches after interruption')
    args = p.parse_args()
    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text())
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    reports, identities, patient_splits = [], set(), {}
    for record in data['records']:
        if record['phase'] not in ['ED', 'ES'] or record['split'] not in ['train', 'val', 'test']:
            raise ValueError('Invalid phase or split')
        key = (record['case_id'], record['phase'])
        if key in identities:
            raise ValueError('Duplicate patient/phase entry')
        identities.add(key)
        if patient_splits.setdefault(record['case_id'], record['split']) != record['split']:
            raise ValueError('Patient split leakage')
        name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in record['case_id'])
        target = root/record['split']/f'{name}_{record["phase"]}.npz'
        record_sha = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        signature = dict(record_sha256=record_sha, seed=args.seed, project_parallel=args.project_parallel, schema=SCHEMA)
        signature_path = target.with_suffix('.input.json')
        if target.exists() and args.resume:
            from core.data import load_case, require_shell
            if not signature_path.exists() or json.loads(signature_path.read_text()) != signature:
                raise ValueError(f'Existing cache was generated with different inputs/settings: {target}')
            require_shell(load_case(target))
            reports.append(dict(file=str(target), resumed=True)); continue
        # Stable per-case RNG; resuming or manifest ordering does not change a pool.
        entropy = int(hashlib.sha256(f'{args.seed}:{key}'.encode()).hexdigest()[:16], 16)
        reports.append(convert(record, target, np.random.default_rng(entropy), args.project_parallel))
        signature_path.write_text(json.dumps(signature, indent=2))
        print(f'{len(reports)}/{len(data["records"])} {target}', flush=True)
    (root/'preparation_report.json').write_text(json.dumps(dict(schema=SCHEMA,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), records=reports), indent=2))


if __name__ == '__main__':
    main()
