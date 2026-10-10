"""Convert neutral aligned contours to NPZ; adapt_static_cache.py supplies them.
The existing physical alignment is preserved; numerical plane errors are audited.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from core.data import polygon_sdf, resample_contour


def convert(record, output, rng, project_parallel=False):
    planes = record['planes']
    if len(planes) < 2:
        raise ValueError('Need >=2 observed contours')
    normals, world_boundaries = [], []
    for plane in planes:
        origin = np.asarray(plane['origin_mm'], float)
        u, v = np.asarray(plane['u'], float), np.asarray(plane['v'], float)
        if max(abs(np.linalg.norm(u)-1), abs(np.linalg.norm(v)-1), abs(u @ v)) > 1e-4:
            raise ValueError('u,v must be orthonormal; pixel spacing belongs in contour_uv_mm')
        uv = np.asarray(plane['contour_uv_mm'], float)
        if len(uv) > 3 and np.linalg.norm(uv[0] - uv[-1]) < 1e-6:
            uv = uv[:-1]
        # adapt_static_cache.py provides CCW polygons with a common angular start.
        bc = resample_contour(uv, 256)
        normals.append(np.cross(u, v))
        world_boundaries.append(origin + bc[:, :1] * u + bc[:, 1:2] * v)
        plane['_uv'] = uv; plane['_bc'] = bc
    axis = normals[0]
    oriented = [n if n @ axis > 0 else -n for n in normals]
    axis = np.mean(oriented, axis=0); axis /= np.linalg.norm(axis)
    center = np.concatenate(world_boundaries).mean(0)
    scale = 100.0  # Fixed millimetres per normalized coordinate, never case bbox scale.
    prepared, max_projection = [], 0.0
    for plane, boundary_mm in zip(planes, world_boundaries):
        origin = np.asarray(plane['origin_mm'], float)
        u, v = np.asarray(plane['u'], float), np.asarray(plane['v'], float)
        uv, bc = plane['_uv'], plane['_bc']
        positions = bc[rng.integers(len(bc), size=768)]
        sigma = rng.choice([1., 3., 6.], size=768, p=[0.4, 0.4, 0.2])
        point_uv = positions + rng.normal(size=positions.shape) * sigma[:, None]
        boxmin, boxmax = uv.min(0) - 10, uv.max(0) + 10
        point_uv = np.r_[point_uv, rng.uniform(boxmin, boxmax, (256, 2))]
        point_mm = origin + point_uv[:, :1] * u + point_uv[:, 1:2] * v
        s_mm = (origin - center) @ axis
        shift_q = ((point_mm - center) @ axis - s_mm)[:, None] * axis
        shift_b = ((boundary_mm - center) @ axis - s_mm)[:, None] * axis
        projection = max(np.linalg.norm(shift_q, axis=1).max(), np.linalg.norm(shift_b, axis=1).max())
        max_projection = max(max_projection, float(projection))
        if projection > 0.01 and not project_parallel:
            raise ValueError(f'Nonparallel stack needs explicit approximation: {projection:.4f} mm')
        point_mm = point_mm - shift_q; boundary_mm = boundary_mm - shift_b
        # If projected, compute plane SDF to the PROJECTED contour in an orthonormal
        # common plane basis, rather than retaining a distorted original distance.
        anchor = np.eye(3)[np.argmin(np.abs(axis))]
        eu = np.cross(anchor, axis); eu /= np.linalg.norm(eu)
        ev = np.cross(axis, eu)
        projected_polygon = origin + uv[:, :1] * u + uv[:, 1:2] * v
        polygon_2d = np.stack([(projected_polygon-origin) @ eu, (projected_polygon-origin) @ ev], -1)
        point_2d = np.stack([(point_mm-origin) @ eu, (point_mm-origin) @ ev], -1)
        sdf = polygon_sdf(point_2d, polygon_2d) / scale
        prepared.append((s_mm / scale, (point_mm-center)/scale, sdf, (boundary_mm-center)/scale))
    prepared.sort(key=lambda x: x[0])
    s, q, sdf, boundary = zip(*prepared)
    if np.any(np.diff(s) <= 1e-5):
        raise ValueError('Duplicate planes; resolve in adapter, never merge silently')
    np.savez_compressed(output, q=np.asarray(q, np.float32), sdf2d=np.asarray(sdf, np.float32),
                        boundary=np.asarray(boundary, np.float32), s=np.asarray(s, np.float32),
                        axis=np.asarray(axis, np.float32), center_mm=center, scale_mm=np.array(scale),
                        phase=np.array(0 if record['phase'] == 'ED' else 1), case_id=np.array(record['case_id']),
                        anatomy=np.array(record.get('anatomy', 'single_filled_region')),
                        provenance_json=np.array(json.dumps({k:v for k,v in record.items() if k!='planes'})))
    return {'file': str(output), 'max_plane_projection_mm': max_projection, 'explicit_parallel_projection': project_parallel}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--project-parallel', action='store_true')
    args = p.parse_args()
    data = json.loads(Path(args.manifest).read_text())
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    reports = []
    rng = np.random.default_rng(20261009)
    identities = set()
    patient_splits = {}
    for record in data['records']:
        if record['phase'] not in ['ED', 'ES'] or record['split'] not in ['train', 'val', 'test']:
            raise ValueError('Invalid phase or split')
        # Keep raw patient id as metadata; filename sanitization is not identity.
        key = (record['case_id'], record['phase'])
        if key in identities:
            raise ValueError('Duplicate patient/phase manifest entry')
        identities.add(key)
        previous = patient_splits.setdefault(record['case_id'], record['split'])
        if previous != record['split']:
            raise ValueError(f'Patient split leakage: {record["case_id"]}')
        name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in record['case_id'])
        folder = root / record['split']; folder.mkdir(exist_ok=True)
        target = folder / f'{name}_{record["phase"]}.npz'
        if target.exists():
            raise ValueError(f'Output already exists or collides: {target}')
        reports.append(convert(record, target, rng, args.project_parallel))
    (root / 'preparation_report.json').write_text(json.dumps(reports, indent=2))
    print(f'Converted {len(reports)} observed phase samples')


if __name__ == '__main__':
    main()
